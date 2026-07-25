import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from libs.chunking.pipeline import chunk_filing_from_source
from libs.core.config import embedding_settings, vectorstore_settings
from libs.core.models import Filing
from libs.generation.generator import Claim, GeneratedAnswer
from libs.generation.pipeline import answer
from libs.indexing.pipeline import index_filing

_FIXTURES_DIR = Path(__file__).parents[2] / "data" / "fixtures"
_DIM = 32


def _deterministic_vector(text: str) -> list[float]:
    rng = random.Random(text)
    return [rng.uniform(-1, 1) for _ in range(_DIM)]


class _FakeChatCompletions:
    """Ignores the real prompt content and returns scripted, plausible
    responses -- this test exercises real retrieval/indexing plumbing end
    to end, not the LLM's actual reasoning (which the mocked-unit tests in
    test_generator.py / test_groundedness.py already cover in isolation).
    """

    def __init__(self, generated: GeneratedAnswer, verdict_for_claim):
        self._generated = generated
        self._verdict_for_claim = verdict_for_claim
        self.calls: list[dict] = []

    def parse(self, *, messages, response_format, **kwargs):
        self.calls.append({"messages": messages, "response_format": response_format})
        if response_format is GeneratedAnswer:
            parsed = self._generated
        else:
            # The groundedness judge call: the claim text is embedded in
            # the user message, so recover which claim this is for.
            user_message = messages[-1]["content"]
            claim = next(c for c in self._generated.claims if c.text in user_message)
            parsed = self._verdict_for_claim(claim)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


class _FakeEmbeddings:
    def create(self, model, input, dimensions):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=_deterministic_vector(text)) for text in input]
        )


class _FakeOpenAIClient:
    def __init__(self, generated: GeneratedAnswer, verdict_for_claim):
        self.embeddings = _FakeEmbeddings()
        self.chat = SimpleNamespace(
            completions=_FakeChatCompletions(generated, verdict_for_claim)
        )


class _FakeCohereClient:
    """Word-overlap scoring, same stand-in used in tests/retrieval/test_pipeline.py."""

    def rerank(self, *, model, query, documents, top_n):
        query_words = set(query.lower().split())
        scored = [
            (i, float(len(query_words & set(doc.lower().split())))) for i, doc in enumerate(documents)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=i, relevance_score=score) for i, score in scored[:top_n]
            ]
        )


class _Verdict:
    def __init__(self, supported: bool, confidence: float):
        self.supported = supported
        self.confidence = confidence


@pytest.fixture(scope="module")
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)


@pytest.fixture
def indexed_amazon_filing(monkeypatch, sparse_model) -> QdrantClient:
    monkeypatch.setattr(embedding_settings, "dimensions", _DIM)
    qdrant_client = QdrantClient(":memory:")

    filing = Filing(
        filing_id="amzn-2025-10k",
        company="Amazon.com, Inc.",
        cik="0001018724",
        fiscal_year=2025,
        source_path=str(_FIXTURES_DIR / "amzn-20251231.html"),
    )
    chunks = chunk_filing_from_source(filing)
    index_filing(chunks, filing, _FakeOpenAIClient(None, None), qdrant_client, sparse_model)
    return qdrant_client


def test_answer_produces_a_grounded_cited_answer_for_a_real_query(
    indexed_amazon_filing, sparse_model
):
    """End-to-end through real retrieval/indexing plumbing (real chunks,
    real Qdrant RRF fusion, real BM25 sparse) with the LLM stages faked to
    always agree the retrieved content supports the query -- verifies the
    pipeline correctly wires retrieve -> generate -> groundedness -> a
    citation-complete answer, using an actual Item 7/8 chunk_id it can
    only have gotten from real retrieval.
    """
    # Discover a real, retrievable chunk_id up front so the scripted
    # GeneratedAnswer cites something that will actually be among the
    # retrieved results, not an arbitrary guess.
    from libs.retrieval.pipeline import retrieve

    candidates = retrieve(
        "North America net sales",
        indexed_amazon_filing,
        _FakeOpenAIClient(None, None),
        sparse_model,
        _FakeCohereClient(),
        top_k=3,
    )
    assert candidates, "expected at least one real retrieval candidate to build the test on"
    target_chunk_id = candidates[0].payload["chunk_id"]
    target_text = candidates[0].payload["text"]

    generated = GeneratedAnswer(
        answer="Net sales figures are reported by segment.",
        claims=[Claim(text="Net sales figures are reported by segment.", chunk_id=target_chunk_id)],
    )
    openai_client = _FakeOpenAIClient(
        generated, verdict_for_claim=lambda claim: _Verdict(True, 0.9)
    )

    result = answer(
        "North America net sales",
        indexed_amazon_filing,
        openai_client,
        sparse_model,
        _FakeCohereClient(),
        top_k=3,
    )

    assert result.answered is True
    assert result.citations[0]["chunk_id"] == target_chunk_id
    assert result.citations[0]["company"] == "Amazon.com, Inc."
    assert result.citations[0]["text"] == target_text
    assert result.citations[0]["section_item_code"] in ("7", "8")


def test_answer_refuses_when_llm_cites_a_chunk_never_retrieved(
    indexed_amazon_filing, sparse_model
):
    """A hallucinated citation must cause a refusal even when real
    retrieval succeeded and found genuinely relevant content.
    """
    generated = GeneratedAnswer(
        answer="Revenue grew significantly.",
        claims=[Claim(text="Revenue grew significantly.", chunk_id="amzn-2025-10k:not-a-real-chunk")],
    )
    openai_client = _FakeOpenAIClient(
        generated, verdict_for_claim=lambda claim: _Verdict(True, 0.9)
    )

    result = answer(
        "North America net sales",
        indexed_amazon_filing,
        openai_client,
        sparse_model,
        _FakeCohereClient(),
        top_k=3,
    )

    assert result.answered is False
    assert result.groundedness.claim_verdicts[0].reason == "cited chunk_id was not among the retrieved sources"
