from types import SimpleNamespace

from libs.core.config import retrieval_settings
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
