"""Tests the API's own responsibilities -- request validation, dependency
wiring, response serialization -- not pipeline correctness, which is
already covered by tests/generation and tests/eval. `answer()` itself is
monkeypatched, same as `tests/eval/test_harness.py` fakes `answer` rather
than re-verifying the pipeline underneath it.
"""
from fastapi.testclient import TestClient

from libs.api import app as app_module
from libs.api.app import app
from libs.generation.pipeline import AnswerResult

client = TestClient(app)


def _override_all_dependencies():
    app.dependency_overrides[app_module.get_openai_client] = lambda: None
    app.dependency_overrides[app_module.get_qdrant_client] = lambda: None
    app.dependency_overrides[app_module.get_sparse_model] = lambda: None
    app.dependency_overrides[app_module.get_cohere_client] = lambda: None


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_answer_endpoint_returns_a_grounded_answer_as_json(monkeypatch):
    def fake_answer(query_text, qdrant_client, openai_client, sparse_model, cohere_client, top_k):
        return AnswerResult(
            query=query_text,
            answered=True,
            answer="Revenue was $1B.",
            citations=[{"chunk_id": "acme-10k:7:0"}],
        )

    monkeypatch.setattr(app_module, "answer", fake_answer)
    _override_all_dependencies()
    try:
        response = client.post("/answer", json={"query": "What was revenue?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert body["answer"] == "Revenue was $1B."
    assert body["citations"] == [{"chunk_id": "acme-10k:7:0"}]


def test_answer_endpoint_returns_a_refusal_as_json(monkeypatch):
    def fake_answer(query_text, qdrant_client, openai_client, sparse_model, cohere_client, top_k):
        return AnswerResult(
            query=query_text,
            answered=False,
            refusal_reason="No relevant sources were found for this question.",
        )

    monkeypatch.setattr(app_module, "answer", fake_answer)
    _override_all_dependencies()
    try:
        response = client.post("/answer", json={"query": "What color is the sky on Mars?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is False
    assert body["answer"] is None
    assert body["refusal_reason"] == "No relevant sources were found for this question."


def test_answer_endpoint_passes_top_k_through_to_the_pipeline(monkeypatch):
    captured = {}

    def fake_answer(query_text, qdrant_client, openai_client, sparse_model, cohere_client, top_k):
        captured["top_k"] = top_k
        return AnswerResult(query=query_text, answered=False, refusal_reason="n/a")

    monkeypatch.setattr(app_module, "answer", fake_answer)
    _override_all_dependencies()
    try:
        client.post("/answer", json={"query": "q", "top_k": 3})
    finally:
        app.dependency_overrides.clear()

    assert captured["top_k"] == 3


def test_answer_endpoint_top_k_defaults_to_none_when_omitted(monkeypatch):
    captured = {}

    def fake_answer(query_text, qdrant_client, openai_client, sparse_model, cohere_client, top_k):
        captured["top_k"] = top_k
        return AnswerResult(query=query_text, answered=False, refusal_reason="n/a")

    monkeypatch.setattr(app_module, "answer", fake_answer)
    _override_all_dependencies()
    try:
        client.post("/answer", json={"query": "q"})
    finally:
        app.dependency_overrides.clear()

    assert captured["top_k"] is None


def test_answer_endpoint_rejects_a_request_missing_the_query_field():
    """FastAPI resolves dependencies alongside body validation rather than
    short-circuiting on a validation error first, so real (uninjected)
    dependencies would try to construct real clients here too -- override
    them even though this test only cares about the 422.
    """
    _override_all_dependencies()
    try:
        response = client.post("/answer", json={})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
