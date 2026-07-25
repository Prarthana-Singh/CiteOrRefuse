import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from libs.chunking.pipeline import chunk_filing_from_source
from libs.core.config import embedding_settings, vectorstore_settings
from libs.core.models import Filing
from libs.indexing.pipeline import index_filing
from libs.retrieval.pipeline import retrieve

_FIXTURES_DIR = Path(__file__).parents[2] / "data" / "fixtures"
_DIM = 32


def _deterministic_vector(text: str) -> list[float]:
    rng = random.Random(text)
    return [rng.uniform(-1, 1) for _ in range(_DIM)]


class _FakeEmbeddings:
    def create(self, model, input, dimensions):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=_deterministic_vector(text)) for text in input]
        )


class _FakeOpenAIClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


class _FakeCohereClient:
    """Scores by query/document word overlap -- a cheap, deterministic
    stand-in for real semantic reranking, per the hard test-isolation rule
    (no real Cohere key in the automated suite). Still exercises the real
    plumbing: real chunk text, real BM25 sparse retrieval, real fusion.
    """

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
    index_filing(chunks, filing, _FakeOpenAIClient(), qdrant_client, sparse_model)
    return qdrant_client


def test_retrieve_surfaces_item7_content_for_a_known_query(indexed_amazon_filing, sparse_model):
    """Mirrors the real live smoke test's query -- a hybrid dense+BM25+RRF
    retrieval for "North America net sales" should surface Item 7 (MD&A)
    content, with a payload that reconstructs a full citation.
    """
    results = retrieve(
        "North America net sales",
        indexed_amazon_filing,
        _FakeOpenAIClient(),
        sparse_model,
        _FakeCohereClient(),
        top_k=5,
    )

    assert len(results) > 0
    assert results[0].score >= results[-1].score  # descending relevance order
    item_codes = {r.payload["section_item_code"] for r in results}
    assert "7" in item_codes or "8" in item_codes

    top = results[0]
    assert top.payload["company"] == "Amazon.com, Inc."
    assert top.payload["part"] in ("I", "II", "III", "IV")
    assert top.payload["char_end"] > top.payload["char_start"]
    assert len(top.payload["text"]) > 0


def test_retrieve_respects_top_k(indexed_amazon_filing, sparse_model):
    results = retrieve(
        "North America net sales",
        indexed_amazon_filing,
        _FakeOpenAIClient(),
        sparse_model,
        _FakeCohereClient(),
        top_k=2,
    )

    assert len(results) == 2
