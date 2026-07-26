"""Orchestrates hybrid retrieval -> rerank for a single query."""
import cohere
from fastembed import SparseTextEmbedding
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient

from libs.core.config import retrieval_settings
from libs.core.logging import get_logger
from libs.retrieval.hybrid_search import hybrid_search
from libs.retrieval.reranker import rerank

logger = get_logger(__name__)


class RetrievedResult(BaseModel):
    """One final, reranked retrieval result. `payload` carries everything
    needed to display a citation (company, part, item, section, char
    offsets, text) without a second lookup -- see
    `libs.vectorstore.qdrant_store._build_payload`."""

    score: float
    payload: dict


def retrieve(
    query_text: str,
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
    sparse_model: SparseTextEmbedding,
    cohere_client: cohere.ClientV2,
    top_k: int | None = None,
    filing_ids: list[str] | None = None,
) -> list[RetrievedResult]:
    """Hybrid dense+BM25 retrieval (Qdrant RRF fusion) -> Cohere rerank ->
    top-K, in descending relevance order.

    `filing_ids`, when given, restricts retrieval to just those filings --
    see `libs.retrieval.hybrid_search`. Defaults to None (no restriction,
    search the whole collection), the same behavior this function has
    always had -- purely additive.
    """
    top_k = top_k or retrieval_settings.default_top_k
    candidates = hybrid_search(
        query_text, qdrant_client, openai_client, sparse_model, filing_ids=filing_ids
    )
    if not candidates:
        logger.info("Retrieval for %r returned no candidates", query_text)
        return []

    documents = [candidate.payload["text"] for candidate in candidates]
    reranked = rerank(query_text, documents, cohere_client, top_n=top_k)

    results = [
        RetrievedResult(score=r.relevance_score, payload=candidates[r.index].payload)
        for r in reranked
    ]
    logger.info(
        "Retrieval for %r: %d fused candidates -> %d reranked results",
        query_text,
        len(candidates),
        len(results),
    )
    return results
