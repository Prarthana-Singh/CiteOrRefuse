"""Wraps Qdrant: a single collection holding both dense and BM25 sparse
vectors per point, rather than two separate retrieval systems."""
import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from libs.core.config import embedding_settings, vectorstore_settings
from libs.core.models import Chunk, Filing

# Fixed, arbitrary namespace UUID -- only needs to be *stable*, not
# meaningful, since it's just an input to uuid5's deterministic hash.
_POINT_ID_NAMESPACE = uuid.UUID("a15e5b1e-6b1d-4c1f-9c0a-9f6b0f7a5f10")


def point_id_for(chunk: Chunk) -> str:
    """Deterministic Qdrant point ID derived from `chunk_id`.

    `chunk_id` is already unique per chunk within (and across, via its
    `filing_id` prefix) filings by construction from the chunking pipeline,
    so hashing it deterministically means re-running ingestion on an
    already-indexed filing overwrites its existing points on upsert instead
    of creating duplicates -- Qdrant's auto-generated IDs would not have
    this property.
    """
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk.chunk_id))


def ensure_collection(client: QdrantClient) -> None:
    """Creates the shared dense+sparse collection if it doesn't exist yet."""
    name = vectorstore_settings.collection_name
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            vectorstore_settings.dense_vector_name: models.VectorParams(
                size=embedding_settings.dimensions,
                distance=models.Distance(vectorstore_settings.dense_distance),
            ),
        },
        sparse_vectors_config={
            vectorstore_settings.sparse_vector_name: models.SparseVectorParams(),
        },
    )


def _build_payload(chunk: Chunk, filing: Filing) -> dict:
    """Everything needed to reconstruct a full citation from a retrieved
    point alone -- no second lookup against the filing/section tables."""
    return {
        "chunk_id": chunk.chunk_id,
        "filing_id": chunk.filing_id,
        "company": filing.company,
        "form_type": filing.form_type,
        "filing_date": filing.filing_date,
        "section_item_code": chunk.section_item_code,
        "section_title": chunk.section_title,
        "part": chunk.part,
        "chunk_type": chunk.chunk_type.value,
        "order_index": chunk.order_index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "token_count": chunk.token_count,
        "text": chunk.text,
    }


def upsert_chunks(
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    filing: Filing,
    client: QdrantClient,
    sparse_model: SparseTextEmbedding,
) -> int:
    """Computes BM25 sparse vectors for `chunks` and upserts dense+sparse
    points with deterministic IDs and a citation-complete payload.

    `dense_vectors` must already be computed (via `libs.embedding`) and be
    in the same order as `chunks`. Calling this twice for the same filing's
    chunks overwrites the existing points rather than duplicating them.
    """
    if not chunks:
        return 0

    ensure_collection(client)
    sparse_vectors = list(sparse_model.embed([chunk.text for chunk in chunks]))

    points = [
        models.PointStruct(
            id=point_id_for(chunk),
            vector={
                vectorstore_settings.dense_vector_name: dense_vector,
                vectorstore_settings.sparse_vector_name: models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                ),
            },
            payload=_build_payload(chunk, filing),
        )
        for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors)
    ]

    client.upsert(collection_name=vectorstore_settings.collection_name, points=points)
    return len(points)


def search_dense(
    client: QdrantClient, query_vector: list[float], top_k: int = 5
) -> list[models.ScoredPoint]:
    """Dense-only similarity search, enough to smoke-test embed -> store ->
    retrieve end-to-end. Hybrid dense+BM25 fusion is query-pipeline scope,
    not this phase's.
    """
    response = client.query_points(
        collection_name=vectorstore_settings.collection_name,
        query=query_vector,
        using=vectorstore_settings.dense_vector_name,
        limit=top_k,
        with_payload=True,
    )
    return response.points
