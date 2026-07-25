"""One-off debug helper: indexes a single fixture and answers a single
query, printing the exact cited chunk_id(s) -- minimal API usage, for
investigating a specific eval-case failure without re-running the whole
eval set against a rate-limited trial key.

Usage: python scripts/debug_single_query.py
"""
import cohere
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

from libs.chunking.pipeline import chunk_filing_from_source
from libs.core.config import vectorstore_settings
from libs.core.models import Filing
from libs.generation.pipeline import answer
from libs.indexing.pipeline import index_filing

QUERY = "What were Apple's total net sales for the three months ended March 28, 2026?"


def main() -> None:
    filing = Filing(
        filing_id="aapl-2026-10q",
        company="Apple Inc.",
        cik="0000320193",
        fiscal_year=2026,
        form_type="10-Q",
        source_path="data/fixtures/aapl-20260328.html",
    )
    chunks = chunk_filing_from_source(filing)

    openai_client = OpenAI()
    qdrant_client = QdrantClient(":memory:")
    sparse_model = SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)
    cohere_client = cohere.ClientV2()

    index_filing(chunks, filing, openai_client, qdrant_client, sparse_model)

    result = answer(QUERY, qdrant_client, openai_client, sparse_model, cohere_client)
    print(f"answered={result.answered}")
    print(f"answer={result.answer}")
    print("citations:")
    for c in result.citations or []:
        print(f"  chunk_id={c['chunk_id']}")
        print(f"  text={c['text'][:300]}")


if __name__ == "__main__":
    main()
