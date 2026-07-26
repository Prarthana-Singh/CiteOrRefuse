"""Tests for `POST /ingest`: a valid upload succeeds and becomes queryable
via `/answer`; the existing `UnsupportedDocumentTypeError` guard rejects a
real 10-Q cleanly; oversized and malformed uploads are rejected before (or
without) doing real ingestion work; the per-client rate limit engages.

No real API key is needed anywhere here: dense embeddings and chat
completions are faked (same combined-fake pattern as
tests/generation/test_pipeline_integration.py), BM25 sparse embedding uses
the real local fastembed model (no network call), and Qdrant runs
in-memory.
"""
import random
from pathlib import Path
from types import SimpleNamespace

from fastembed import SparseTextEmbedding
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from libs.api import app as app_module
from libs.api.app import app
from libs.api.rate_limit import FixedWindowRateLimiter
from libs.core.config import embedding_settings, vectorstore_settings
from libs.generation.generator import Claim, GeneratedAnswer

client = TestClient(app)

_FIXTURES_DIR = Path(__file__).parents[2] / "data" / "fixtures"
_SAMPLE_10K = _FIXTURES_DIR / "sample_10k_excerpt.htm"
_APPLE_10Q = _FIXTURES_DIR / "aapl-20260328.html"
_DIM = 32


def _deterministic_vector(text: str) -> list[float]:
    rng = random.Random(text)
    return [rng.uniform(-1, 1) for _ in range(_DIM)]


class _FakeChatCompletions:
    """Ignores prompt content; scripted to always agree the retrieved
    content supports whatever claim it's given -- these tests exercise
    ingest-then-retrieve plumbing, not the LLM's actual reasoning (covered
    elsewhere, see tests/generation)."""

    def __init__(self, generated: GeneratedAnswer | None = None):
        self._generated = generated

    def parse(self, *, messages, response_format, **kwargs):
        if response_format is GeneratedAnswer:
            parsed = self._generated
        else:
            parsed = SimpleNamespace(supported=True, confidence=0.95)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


class _FakeEmbeddings:
    def create(self, model, input, dimensions):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=_deterministic_vector(text)) for text in input]
        )


class _FakeOpenAIClient:
    def __init__(self, generated: GeneratedAnswer | None = None):
        self.embeddings = _FakeEmbeddings()
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(generated))


class _FakeCohereClient:
    def rerank(self, *, model, query, documents, top_n):
        query_words = set(query.lower().split())
        scored = [
            (i, float(len(query_words & set(doc.lower().split())))) for i, doc in enumerate(documents)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return SimpleNamespace(
            results=[SimpleNamespace(index=i, relevance_score=score) for i, score in scored[:top_n]]
        )


def _generous_rate_limiter() -> FixedWindowRateLimiter:
    """Each test gets its own limiter override -- the real one is a
    process-wide `lru_cache` singleton (see libs/api/dependencies.py) that
    would otherwise leak call counts between unrelated tests in the same
    pytest run.
    """
    return FixedWindowRateLimiter(max_calls=1000, window_seconds=60.0)


def test_ingest_accepts_a_valid_10k_and_it_becomes_queryable_via_answer(monkeypatch):
    monkeypatch.setattr(embedding_settings, "dimensions", _DIM)
    qdrant_client = QdrantClient(":memory:")
    sparse_model = SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)

    app.dependency_overrides[app_module.get_openai_client] = lambda: _FakeOpenAIClient()
    app.dependency_overrides[app_module.get_qdrant_client] = lambda: qdrant_client
    app.dependency_overrides[app_module.get_sparse_model] = lambda: sparse_model
    app.dependency_overrides[app_module.get_cohere_client] = lambda: _FakeCohereClient()
    app.dependency_overrides[app_module.get_ingest_rate_limiter] = _generous_rate_limiter
    try:
        with open(_SAMPLE_10K, "rb") as f:
            response = client.post(
                "/ingest",
                files={"file": ("sample_10k_excerpt.htm", f, "text/html")},
                data={"company": "Acme Robotics, Inc."},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["company"] == "Acme Robotics, Inc."
        assert body["sections_detected"] > 0
        assert body["chunks_indexed"] > 0
        filing_id = body["filing_id"]

        # Discover a real, retrievable chunk_id from the just-ingested
        # filing to script the fake generator's citation against -- same
        # discovery-then-script pattern as
        # tests/generation/test_pipeline_integration.py.
        from libs.retrieval.pipeline import retrieve

        candidates = retrieve(
            "fiscal 2023 revenue increase",
            qdrant_client,
            _FakeOpenAIClient(),
            sparse_model,
            _FakeCohereClient(),
            filing_ids=[filing_id],
        )
        assert candidates, "expected the just-ingested filing's own chunks to be retrievable"
        assert all(c.payload["filing_id"] == filing_id for c in candidates)
        target_chunk_id = candidates[0].payload["chunk_id"]

        generated = GeneratedAnswer(
            answer="Fiscal 2023 revenue increased 22% year-over-year.",
            claims=[Claim(text="Fiscal 2023 revenue increased 22% year-over-year.", chunk_id=target_chunk_id)],
        )
        app.dependency_overrides[app_module.get_openai_client] = lambda: _FakeOpenAIClient(generated)

        answer_response = client.post(
            "/answer",
            json={"query": "How much did fiscal 2023 revenue increase?", "filing_id": filing_id},
        )
        assert answer_response.status_code == 200
        answer_body = answer_response.json()
        assert answer_body["answered"] is True
        assert answer_body["citations"][0]["chunk_id"] == target_chunk_id
        assert all(cid.startswith(filing_id) for cid in answer_body["retrieved_chunk_ids"])
    finally:
        app.dependency_overrides.clear()


def test_ingest_rejects_a_real_10q_cleanly_via_the_document_type_guard():
    """The endpoint always ingests as form_type="10-K" (no user override),
    so a real 10-Q upload must be rejected by the existing
    UnsupportedDocumentTypeError guard doing real work here, not a crash.
    """
    app.dependency_overrides[app_module.get_openai_client] = lambda: _FakeOpenAIClient()
    app.dependency_overrides[app_module.get_qdrant_client] = lambda: QdrantClient(":memory:")
    app.dependency_overrides[app_module.get_sparse_model] = lambda: SparseTextEmbedding(
        model_name=vectorstore_settings.sparse_model
    )
    app.dependency_overrides[app_module.get_ingest_rate_limiter] = _generous_rate_limiter
    try:
        with open(_APPLE_10Q, "rb") as f:
            response = client.post(
                "/ingest",
                files={"file": ("aapl-20260328.html", f, "text/html")},
                data={"company": "Apple Inc."},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert "10-Q" in response.json()["detail"]


def _override_non_ingest_dependencies():
    """FastAPI resolves every declared `Depends(...)` before running the
    route body, even ones a request will never reach (e.g. a request
    that's rejected on the file extension check before touching Qdrant at
    all) -- same reasoning as `test_app.py`'s equivalent helper.
    """
    app.dependency_overrides[app_module.get_openai_client] = lambda: _FakeOpenAIClient()
    app.dependency_overrides[app_module.get_qdrant_client] = lambda: QdrantClient(":memory:")
    app.dependency_overrides[app_module.get_sparse_model] = lambda: SparseTextEmbedding(
        model_name=vectorstore_settings.sparse_model
    )


def test_ingest_rejects_an_oversized_file_before_parsing(monkeypatch):
    monkeypatch.setattr(app_module.ingest_settings, "max_upload_bytes", 100)
    _override_non_ingest_dependencies()
    app.dependency_overrides[app_module.get_ingest_rate_limiter] = _generous_rate_limiter
    try:
        oversized_content = b"<html><body><p>" + b"a" * 200 + b"</p></body></html>"
        response = client.post(
            "/ingest",
            files={"file": ("big.htm", oversized_content, "text/html")},
            data={"company": "Big Co."},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 413


def test_ingest_rejects_malformed_non_html_content_cleanly():
    """Bare text with no wrapping block-level tags at all yields zero
    parseable blocks -- `ParsingError`, caught and mapped to a 422, not an
    unhandled exception.
    """
    app.dependency_overrides[app_module.get_openai_client] = lambda: _FakeOpenAIClient()
    app.dependency_overrides[app_module.get_qdrant_client] = lambda: QdrantClient(":memory:")
    app.dependency_overrides[app_module.get_sparse_model] = lambda: SparseTextEmbedding(
        model_name=vectorstore_settings.sparse_model
    )
    app.dependency_overrides[app_module.get_ingest_rate_limiter] = _generous_rate_limiter
    try:
        garbage = b"<html><body>not a real SEC filing at all, just garbage</body></html>"
        response = client.post(
            "/ingest",
            files={"file": ("garbage.htm", garbage, "text/html")},
            data={"company": "Nobody Inc."},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert "No content blocks" in response.json()["detail"]


def test_ingest_rejects_a_non_html_file_extension():
    _override_non_ingest_dependencies()
    app.dependency_overrides[app_module.get_ingest_rate_limiter] = _generous_rate_limiter
    try:
        response = client.post(
            "/ingest",
            files={"file": ("filing.pdf", b"whatever", "application/pdf")},
            data={"company": "Acme Robotics, Inc."},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400


def test_ingest_requires_a_company_field():
    _override_non_ingest_dependencies()
    app.dependency_overrides[app_module.get_ingest_rate_limiter] = _generous_rate_limiter
    try:
        with open(_SAMPLE_10K, "rb") as f:
            response = client.post("/ingest", files={"file": ("sample_10k_excerpt.htm", f, "text/html")})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_ingest_rate_limit_rejects_calls_beyond_the_window_limit():
    """The override must return the *same* limiter instance on every
    resolution -- FastAPI resolves a plain (non-cached) `Depends(...)`
    fresh per request, so a lambda constructing a new limiter each call
    would never accumulate a call history across requests.
    """
    _override_non_ingest_dependencies()
    limiter = FixedWindowRateLimiter(max_calls=2, window_seconds=60.0)
    app.dependency_overrides[app_module.get_ingest_rate_limiter] = lambda: limiter
    try:
        for _ in range(2):
            response = client.post(
                "/ingest",
                files={"file": ("filing.pdf", b"whatever", "application/pdf")},
                data={"company": "Acme Robotics, Inc."},
            )
            # These fail on the extension check (400), not the rate limit --
            # the point is that the rate limiter itself is checked and
            # counts the call regardless of what happens afterward.
            assert response.status_code == 400

        third_response = client.post(
            "/ingest",
            files={"file": ("filing.pdf", b"whatever", "application/pdf")},
            data={"company": "Acme Robotics, Inc."},
        )
    finally:
        app.dependency_overrides.clear()

    assert third_response.status_code == 429
