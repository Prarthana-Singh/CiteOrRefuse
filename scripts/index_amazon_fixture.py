"""Runs the full Phase 1-3 pipeline end-to-end against the real Amazon
fixture: ingest -> chunk -> embed (OpenAI) -> upsert (Qdrant, dense + BM25
sparse) -> hybrid retrieve (RRF fusion) -> rerank (Cohere) -> print
reconstructed citations for the final top-K.

Requires real OPENAI_API_KEY and COHERE_API_KEY in the environment -- this
hits both live APIs, unlike the test suite (which mocks both). Uses an
in-memory Qdrant instance, so nothing external needs to be running.

Usage: python scripts/index_amazon_fixture.py
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
from libs.retrieval.pipeline import retrieve

FIXTURE_PATH = Path(__file__).parent.parent / "data" / "fixtures" / "amzn-20251231.html"


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
    print(f"  -> {len(chunks)} chunks")

    openai_client = OpenAI()
    qdrant_client = QdrantClient(":memory:")
    print(f"Loading sparse model ({vectorstore_settings.sparse_model})...")
    sparse_model = SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)

    print("Embedding + indexing (this calls the real OpenAI API)...")
    result = index_filing(chunks, filing, openai_client, qdrant_client, sparse_model)
    print(f"  -> {result.points_written} points written for {result.filing_id}")

    cohere_client = cohere.ClientV2()

    query_text = "North America net sales"
    print(f"\nRetrieving (hybrid dense+BM25 RRF -> Cohere rerank) for: {query_text!r}")
    results = retrieve(query_text, qdrant_client, openai_client, sparse_model, cohere_client, top_k=3)

    for i, result in enumerate(results):
        payload = result.payload
        print(
            f"\n[{i}] score={result.score:.4f} "
            f"{payload['company']} {payload['form_type']}, "
            f"Part {payload['part']} Item {payload['section_item_code']} "
            f"({payload['section_title']})"
        )
        print(f"    chars=({payload['char_start']}, {payload['char_end']})")
        preview = payload["text"].replace("\n", " ")[:200]
        print(f"    {preview}...")


if __name__ == "__main__":
    main()
