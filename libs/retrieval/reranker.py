"""Wraps Cohere Rerank: reorders candidates by relevance to a query."""
import time
from typing import NamedTuple

import cohere

from libs.core.config import retrieval_settings
from libs.core.logging import get_logger

logger = get_logger(__name__)

# A Cohere trial key is capped at 10 calls/minute; a single eval-harness run
# across several cases/queries can burst past that easily. Retrying a 429
# with backoff is cheap insurance against a transient rate limit derailing
# an otherwise-working run -- not a workaround for a real quota problem
# (a production key wouldn't hit this in normal usage).
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 7


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

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = cohere_client.rerank(
                model=retrieval_settings.rerank_model,
                query=query_text,
                documents=documents,
                top_n=min(top_n, len(documents)) if top_n is not None else len(documents),
            )
            break
        except cohere.TooManyRequestsError:
            if attempt == _MAX_RETRIES:
                raise
            logger.warning(
                "Cohere rerank rate-limited, retrying in %ds (attempt %d/%d)",
                _RETRY_DELAY_SECONDS,
                attempt + 1,
                _MAX_RETRIES,
            )
            time.sleep(_RETRY_DELAY_SECONDS)
    return [
        RerankedCandidate(index=result.index, relevance_score=result.relevance_score)
        for result in response.results
    ]
