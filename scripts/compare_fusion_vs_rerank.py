"""Diagnostic: shows Qdrant RRF-fusion-only ranking vs Cohere-reranked
ranking for the same query, side by side, to see whether rerank actually
changes the order or just confirms what fusion already had right.

Requires real OPENAI_API_KEY and COHERE_API_KEY. Uses an in-memory Qdrant
instance, so nothing external needs to be running.

Usage: python scripts/compare_fusion_vs_rerank.py
"""
from pathlib import Path

import cohere
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

from libs.chunking.pipeline import chunk_filing_from_source
from libs.core.config import vectorstore_settings
from libs.core.models import Filing
from libs.indexing.pipeline import index_filing
from libs.retrieval.hybrid_search import hybrid_search
from libs.retrieval.reranker import rerank

FIXTURE_PATH = Path(__file__).parent.parent / "data" / "fixtures" / "amzn-20251231.html"
QUERY = "North America net sales"
TOP_N = 5


def _describe(payload: dict) -> str:
    preview = payload["text"].replace("\n", " ")[:120]
    return f"Item {payload['section_item_code']} ({payload['section_title'][:40]}) | {preview}..."


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

    openai_client = OpenAI()
    qdrant_client = QdrantClient(":memory:")
    sparse_model = SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)
    cohere_client = cohere.ClientV2()

    print("Embedding + indexing...")
    index_filing(chunks, filing, openai_client, qdrant_client, sparse_model)

    print(f"\nQuery: {QUERY!r}\n")

    fused = hybrid_search(QUERY, qdrant_client, openai_client, sparse_model)
    print(f"=== Stage 1: RRF fusion only (top-{TOP_N} of {len(fused)} fused candidates) ===")
    for i, point in enumerate(fused[:TOP_N]):
        print(f"[{i}] fusion_score={point.score:.4f}  chunk_id={point.payload['chunk_id']}")
        print(f"     {_describe(point.payload)}")

    documents = [c.payload["text"] for c in fused]
    reranked = rerank(QUERY, documents, cohere_client, top_n=TOP_N)
    print(f"\n=== Stage 2: Cohere rerank (top-{TOP_N}, reranked from all {len(fused)} fused candidates) ===")
    for i, r in enumerate(reranked):
        payload = fused[r.index].payload
        print(
            f"[{i}] rerank_score={r.relevance_score:.4f}  "
            f"was_fusion_rank={r.index}  chunk_id={payload['chunk_id']}"
        )
        print(f"     {_describe(payload)}")

    fusion_top_ids = [p.payload["chunk_id"] for p in fused[:TOP_N]]
    rerank_top_ids = [fused[r.index].payload["chunk_id"] for r in reranked]
    print("\n=== Comparison ===")
    print(f"Fusion top-{TOP_N} chunk_ids: {fusion_top_ids}")
    print(f"Rerank top-{TOP_N} chunk_ids: {rerank_top_ids}")
    print(f"Same set:   {set(fusion_top_ids) == set(rerank_top_ids)}")
    print(f"Same order: {fusion_top_ids == rerank_top_ids}")


if __name__ == "__main__":
    main()
