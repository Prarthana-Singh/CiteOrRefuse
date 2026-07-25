"""Phase 3 close-out diagnostic: verifies the rerank-justification story
with actual per-branch (dense-only, sparse-only) scores, instead of just
inferring from the fused RRF result -- and prints the full text of the
Item 7A chunk that entered rerank's top-5 unprompted, so its relevance can
be judged directly rather than assumed from its score alone.

Requires OPENAI_API_KEY only -- no Cohere client needed, since this
inspects fusion's two input branches, not the rerank stage itself.

Usage: python scripts/verify_rerank_justification.py
"""
from pathlib import Path

from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

from libs.chunking.pipeline import chunk_filing_from_source
from libs.core.config import vectorstore_settings
from libs.core.models import Filing
from libs.indexing.pipeline import index_filing
from libs.retrieval.hybrid_search import embed_query

FIXTURE_PATH = Path(__file__).parent.parent / "data" / "fixtures" / "amzn-20251231.html"
QUERY = "North America net sales"
TARGET_CHUNKS = {
    "amzn-2025-10k:8:table:193": "3-year table (Item 8) -- rerank rank 1",
    "amzn-2025-10k:7:table:61": "2-year table (Item 7) -- rerank rank 2",
}
ITEM_7A_CHUNK_ID = "amzn-2025-10k:7A:78"
BRANCH_SEARCH_LIMIT = 50


def _branch_scores(client: QdrantClient, query, using: str) -> dict[str, float]:
    """Ranked chunk_id -> score for a single vector field, queried alone
    (no prefetch/fusion) -- the raw per-branch signal RRF fuses away."""
    response = client.query_points(
        collection_name=vectorstore_settings.collection_name,
        query=query,
        using=using,
        limit=BRANCH_SEARCH_LIMIT,
        with_payload=False,
    )
    return {str(p.id): p.score for p in response.points}, {
        i + 1: p.id for i, p in enumerate(response.points)
    }


def main() -> None:
    filing = Filing(
        filing_id="amzn-2025-10k",
        company="Amazon.com, Inc.",
        cik="0001018724",
        fiscal_year=2025,
        source_path=str(FIXTURE_PATH),
    )
    print(f"Ingesting + chunking {FIXTURE_PATH.name}...")
    chunks = chunk_filing_from_source(filing)
    chunks_by_id = {c.chunk_id: c for c in chunks}

    openai_client = OpenAI()
    qdrant_client = QdrantClient(":memory:")
    sparse_model = SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)

    print("Embedding + indexing...")
    index_filing(chunks, filing, openai_client, qdrant_client, sparse_model)

    dense_vector, sparse_vector = embed_query(QUERY, openai_client, sparse_model)

    from libs.vectorstore.qdrant_store import point_id_for

    target_point_ids = {
        point_id_for(chunks_by_id[chunk_id]): chunk_id for chunk_id in TARGET_CHUNKS
    }

    dense_scores, dense_ranks = _branch_scores(
        qdrant_client, dense_vector, vectorstore_settings.dense_vector_name
    )
    sparse_scores, sparse_ranks = _branch_scores(
        qdrant_client, sparse_vector, vectorstore_settings.sparse_vector_name
    )
    dense_rank_by_id = {v: k for k, v in dense_ranks.items()}
    sparse_rank_by_id = {v: k for k, v in sparse_ranks.items()}

    print(f"\n=== Task 1: per-branch scores for {QUERY!r} (top-{BRANCH_SEARCH_LIMIT} of each branch) ===")
    for chunk_id, label in TARGET_CHUNKS.items():
        point_id = point_id_for(chunks_by_id[chunk_id])
        d_score = dense_scores.get(point_id)
        s_score = sparse_scores.get(point_id)
        d_rank = dense_rank_by_id.get(point_id, f">{BRANCH_SEARCH_LIMIT}")
        s_rank = sparse_rank_by_id.get(point_id, f">{BRANCH_SEARCH_LIMIT}")
        print(f"\n{label}\n  chunk_id: {chunk_id}")
        print(f"  dense-only:  score={d_score if d_score is not None else 'n/a'}  rank={d_rank}")
        print(f"  sparse-only: score={s_score if s_score is not None else 'n/a'}  rank={s_rank}")

    print(f"\n\n=== Task 2: full text of {ITEM_7A_CHUNK_ID} (entered rerank top-5 at rank 5) ===")
    item_7a_chunk = chunks_by_id[ITEM_7A_CHUNK_ID]
    print(f"Section: Item {item_7a_chunk.section_item_code} -- {item_7a_chunk.section_title}")
    print(f"Part: {item_7a_chunk.part}\n")
    print(item_7a_chunk.text)


if __name__ == "__main__":
    main()
