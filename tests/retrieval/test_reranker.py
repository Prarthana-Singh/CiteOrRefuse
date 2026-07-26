from types import SimpleNamespace

import cohere
import pytest

from libs.core.config import retrieval_settings
from libs.retrieval import reranker as reranker_module
from libs.retrieval.reranker import rerank


class _FakeCohereClient:
    def __init__(self, order: list[tuple[int, float]]):
        """`order` is a list of (original_index, relevance_score) pairs,
        already in the descending order Cohere would return them in."""
        self._order = order
        self.calls: list[dict] = []

    def rerank(self, *, model, query, documents, top_n):
        self.calls.append({"model": model, "query": query, "documents": list(documents), "top_n": top_n})
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=index, relevance_score=score)
                for index, score in self._order[:top_n]
            ]
        )


def test_rerank_reorders_candidates_by_relevance_score():
    documents = ["irrelevant paragraph", "North America net sales grew 12%", "unrelated legal text"]
    # Cohere would rank document 1 highest, then 2, then 0.
    client = _FakeCohereClient(order=[(1, 0.95), (2, 0.4), (0, 0.1)])

    results = rerank("North America net sales", documents, client, top_n=3)

    assert [r.index for r in results] == [1, 2, 0]
    assert results[0].relevance_score == 0.95


def test_rerank_uses_the_configured_model_and_passes_query_through():
    client = _FakeCohereClient(order=[(0, 0.5)])

    rerank("some query", ["doc a"], client, top_n=1)

    assert client.calls[0]["model"] == retrieval_settings.rerank_model
    assert client.calls[0]["query"] == "some query"
    assert client.calls[0]["documents"] == ["doc a"]


def test_rerank_caps_top_n_at_the_number_of_documents():
    client = _FakeCohereClient(order=[(0, 0.9), (1, 0.5)])

    rerank("query", ["doc a", "doc b"], client, top_n=50)

    assert client.calls[0]["top_n"] == 2


def test_rerank_with_no_documents_returns_empty_without_calling_cohere():
    client = _FakeCohereClient(order=[])

    results = rerank("query", [], client, top_n=5)

    assert results == []
    assert client.calls == []


def test_rerank_default_top_n_is_all_documents():
    client = _FakeCohereClient(order=[(0, 0.9), (1, 0.5), (2, 0.2)])

    rerank("query", ["a", "b", "c"], client)

    assert client.calls[0]["top_n"] == 3


class _FlakyCohereClient:
    """Raises `TooManyRequestsError` on the first `fail_count` calls, then
    succeeds -- simulates a trial-tier rate limit that clears after a
    retry."""

    def __init__(self, fail_count: int, headers: dict | None = None):
        self._fail_count = fail_count
        self._headers = headers
        self.calls = 0

    def rerank(self, *, model, query, documents, top_n):
        self.calls += 1
        if self.calls <= self._fail_count:
            raise cohere.TooManyRequestsError(body="rate limited", headers=self._headers)
        return SimpleNamespace(results=[SimpleNamespace(index=0, relevance_score=0.9)])


def test_rerank_retries_on_rate_limit_and_eventually_succeeds(monkeypatch):
    monkeypatch.setattr(reranker_module.time, "sleep", lambda seconds: None)
    client = _FlakyCohereClient(fail_count=2)

    results = rerank("query", ["doc a"], client, top_n=1)

    assert results[0].relevance_score == 0.9
    assert client.calls == 3


def test_rerank_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(reranker_module.time, "sleep", lambda seconds: None)
    client = _FlakyCohereClient(fail_count=reranker_module._MAX_RETRIES + 1)

    with pytest.raises(cohere.TooManyRequestsError):
        rerank("query", ["doc a"], client, top_n=1)

    assert client.calls == reranker_module._MAX_RETRIES + 1


def test_rerank_backs_off_using_server_retry_after_header_when_present(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(reranker_module.time, "sleep", sleep_calls.append)
    client = _FlakyCohereClient(fail_count=1, headers={"Retry-After": "3"})

    rerank("query", ["doc a"], client, top_n=1)

    assert sleep_calls == [3.0]


def test_rerank_falls_back_to_default_backoff_when_no_retry_after_header(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(reranker_module.time, "sleep", sleep_calls.append)
    client = _FlakyCohereClient(fail_count=1, headers=None)

    rerank("query", ["doc a"], client, top_n=1)

    assert sleep_calls == [reranker_module._RETRY_DELAY_SECONDS]
