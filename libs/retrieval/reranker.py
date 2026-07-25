"""Wraps Cohere Rerank: reorders candidates by relevance to a query."""
from typing import NamedTuple

import cohere

from libs.core.config import retrieval_settings


class RerankedCandidate(NamedTuple):
    """`index` refers back to the original position in the `documents`
    list passed to `rerank`, so callers can map a result back to whatever
    richer object (e.g. a Qdrant `ScoredPoint`) that text came from."""

    index: int
    relevance_score: float


def rerank(
    query_text: str,
    documents: list[str],
    cohere_client: cohere.ClientV2,
    top_n: int | None = None,
) -> list[RerankedCandidate]:
    """Reorders `documents` by relevance to `query_text` via Cohere Rerank.

    Returns results in descending relevance order. `cohere_client` is
    injected rather than constructed here so tests never need a real key.
    """
    if not documents:
        return []

    response = cohere_client.rerank(
        model=retrieval_settings.rerank_model,
        query=query_text,
        documents=documents,
        top_n=min(top_n, len(documents)) if top_n is not None else len(documents),
    )
    return [
        RerankedCandidate(index=result.index, relevance_score=result.relevance_score)
        for result in response.results
    ]
