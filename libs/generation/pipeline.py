"""Orchestrates retrieve -> generate -> groundedness-check -> answer or refuse."""
import cohere
from fastembed import SparseTextEmbedding
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient

from libs.core.config import groundedness_settings
from libs.core.logging import get_logger
from libs.generation.context_builder import build_context
from libs.generation.generator import generate_answer
from libs.generation.groundedness import GroundednessResult, check_groundedness
from libs.retrieval.pipeline import retrieve

logger = get_logger(__name__)


class AnswerResult(BaseModel):
    """Either a grounded, cited answer, or a refusal -- never a guess.

    `groundedness` is populated whenever generation actually ran (i.e. not
    on the no-candidates-retrieved refusal path), even when the answer was
    refused, so callers can inspect *why* -- which claim(s) failed, and
    whether it was a hallucinated citation or a judge's low-confidence call.
    """

    query: str
    answered: bool
    answer: str | None = None
    citations: list[dict] | None = None
    refusal_reason: str | None = None
    groundedness: GroundednessResult | None = None


def answer(
    query_text: str,
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
    sparse_model: SparseTextEmbedding,
    cohere_client: cohere.ClientV2,
    top_k: int | None = None,
) -> AnswerResult:
    """The full "cite or refuse" pipeline for a single query."""
    results = retrieve(query_text, qdrant_client, openai_client, sparse_model, cohere_client, top_k)
    if not results:
        return AnswerResult(
            query=query_text,
            answered=False,
            refusal_reason="No relevant sources were found for this question.",
        )

    context = build_context(results)
    generated = generate_answer(query_text, context, openai_client)
    groundedness = check_groundedness(generated, results, openai_client)

    if not groundedness.is_grounded or groundedness.overall_confidence < groundedness_settings.confidence_threshold:
        logger.info(
            "Refusing to answer %r: grounded=%s confidence=%.2f",
            query_text,
            groundedness.is_grounded,
            groundedness.overall_confidence,
        )
        return AnswerResult(
            query=query_text,
            answered=False,
            refusal_reason=groundedness_settings.refusal_message,
            groundedness=groundedness,
        )

    # Citations are built from `groundedness.claim_verdicts`, not
    # `generated.claims` directly: a claim's `chunk_id` may have been
    # corrected by citation-rebind (see `libs.generation.groundedness`), in
    # which case `generated.claims` still holds the original, failed
    # chunk_id -- using it here would cite a source that was never actually
    # verified to support the claim.
    payload_by_chunk_id = {r.payload["chunk_id"]: r.payload for r in results}
    cited_chunk_ids = dict.fromkeys(v.claim.chunk_id for v in groundedness.claim_verdicts)
    citations = [payload_by_chunk_id[chunk_id] for chunk_id in cited_chunk_ids]

    logger.info(
        "Answered %r: %d claims, %d unique citations, confidence=%.2f, rebinds=%d",
        query_text,
        len(generated.claims),
        len(citations),
        groundedness.overall_confidence,
        groundedness.rebind_count,
    )
    return AnswerResult(
        query=query_text,
        answered=True,
        answer=generated.answer,
        citations=citations,
        groundedness=groundedness,
    )
