"""Orchestrates embedding -> upsert for one filing's chunks."""
from fastembed import SparseTextEmbedding
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient

from libs.core.logging import get_logger
from libs.core.models import Chunk, Filing
from libs.embedding.openai_embedder import embed_chunks
from libs.vectorstore.qdrant_store import upsert_chunks

logger = get_logger(__name__)


class IndexResult(BaseModel):
    """Outcome of indexing one filing's chunks."""

    filing_id: str
    points_written: int


def index_filing(
    chunks: list[Chunk],
    filing: Filing,
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
    sparse_model: SparseTextEmbedding,
) -> IndexResult:
    """Embeds (dense, via OpenAI) and upserts (dense + BM25 sparse, via
    Qdrant) one filing's chunks.

    Idempotent: re-running against an already-indexed filing's chunks
    overwrites the existing points (see
    `libs.vectorstore.qdrant_store.point_id_for`) rather than duplicating
    them, so this is safe to call again after a partial failure or a
    source-filing update.
    """
    dense_vectors = embed_chunks(chunks, filing, openai_client)
    points_written = upsert_chunks(chunks, dense_vectors, filing, qdrant_client, sparse_model)
    logger.info(
        "Indexed filing %s: %d points written",
        filing.filing_id,
        points_written,
    )
    return IndexResult(filing_id=filing.filing_id, points_written=points_written)
