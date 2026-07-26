"""Hybrid dense+BM25 retrieval via Qdrant's native RRF fusion.

Server-side (works in Qdrant's `:memory:` local mode too, so no real
Qdrant instance is needed for tests): prefetch top candidates from the
dense and sparse vector fields independently, then fuse with reciprocal
rank fusion. This is deliberately not hand-rolled -- same reasoning as
Phase 2's choice to use fastembed's BM25 rather than a hand-rolled index.
"""
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient, models

from libs.core.config import embedding_settings, retrieval_settings, vectorstore_settings


def embed_query(
    query_text: str, openai_client: OpenAI, sparse_model: SparseTextEmbedding
) -> tuple[list[float], models.SparseVector]:
    """Embeds a raw query both dense and sparse.

    Unlike chunks (see `libs.embedding.openai_embedder.build_embedding_text`),
    a query gets no context header prepended -- that header exists to
    anchor otherwise-bare content (e.g. a table of numbers) topically, and
    a natural-language query is already semantically complete on its own.
    """
    dense_response = openai_client.embeddings.create(
        model=embedding_settings.model,
        input=[query_text],
        dimensions=embedding_settings.dimensions,
    )
    dense_vector = dense_response.data[0].embedding

    sparse_embedding = next(iter(sparse_model.embed([query_text])))
    sparse_vector = models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )
    return dense_vector, sparse_vector


def _filing_id_filter(filing_ids: list[str] | None) -> models.Filter | None:
    """Restricts search to points whose `filing_id` payload field is in
    `filing_ids`, or no restriction at all when `filing_ids` is None --
    callers that don't care about scoping (every caller before the
    upload-a-filing endpoint existed) get identical behavior to before
    this parameter was added.
    """
    if not filing_ids:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="filing_id", match=models.MatchAny(any=filing_ids))]
    )


def hybrid_search(
    query_text: str,
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
    sparse_model: SparseTextEmbedding,
    limit: int | None = None,
    filing_ids: list[str] | None = None,
) -> list[models.ScoredPoint]:
    """Runs a Qdrant-native RRF-fused query over dense+BM25 prefetch
    results, returning fused candidates with full payload.

    `filing_ids`, when given, restricts results to just those filings --
    used to scope a query to one just-uploaded filing without it being
    mixed with (or, from the other direction, without an uploaded filing
    leaking into) other content in the same shared collection.

    Not yet reranked -- see `libs.retrieval.reranker` for the Cohere
    rerank stage that turns these into a final answer-ready ordering.
    """
    dense_vector, sparse_vector = embed_query(query_text, openai_client, sparse_model)
    query_filter = _filing_id_filter(filing_ids)

    response = qdrant_client.query_points(
        collection_name=vectorstore_settings.collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=vectorstore_settings.dense_vector_name,
                limit=retrieval_settings.prefetch_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=vectorstore_settings.sparse_vector_name,
                limit=retrieval_settings.prefetch_limit,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit or retrieval_settings.candidate_limit,
        with_payload=True,
    )
    return response.points
