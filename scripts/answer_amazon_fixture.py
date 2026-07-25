"""Runs the full CiteOrRefuse pipeline end-to-end against the real Amazon
fixture: ingest -> chunk -> embed -> index -> retrieve -> generate ->
groundedness-check -> answer or refuse.

Includes one query the filing should be able to answer, and one it should
refuse -- exercising both halves of "cite or refuse", not just the happy
path.

Requires real OPENAI_API_KEY and COHERE_API_KEY. Uses an in-memory Qdrant
instance, so nothing external needs to be running.

Usage: python scripts/answer_amazon_fixture.py
"""
from pathlib import Path

import cohere
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

from libs.chunking.pipeline import chunk_filing_from_source
from libs.core.config import vectorstore_settings
from libs.core.models import Filing
from libs.generation.pipeline import answer
from libs.indexing.pipeline import index_filing

FIXTURE_PATH = Path(__file__).parent.parent / "data" / "fixtures" / "amzn-20251231.html"

QUERIES = [
    "What were North America net sales in 2025?",
    "What color is the sky on Mars?",  # not in any 10-K -- should refuse
]


def _print_result(result) -> None:
    if result.answered:
        print(f"ANSWERED (confidence={result.groundedness.overall_confidence:.2f}):")
        print(f"  {result.answer}")
        print(f"\n  Citations ({len(result.citations)}):")
        for c in result.citations:
            print(
                f"   - [{c['chunk_id']}] {c['company']} {c['form_type']}, "
                f"Part {c['part']} Item {c['section_item_code']} ({c['section_title']})"
            )
            preview = c["text"].replace("\n", " ")[:150]
            print(f"     {preview}...")
    else:
        print(f"REFUSED: {result.refusal_reason}")
        if result.groundedness:
            print(
                f"  (grounded={result.groundedness.is_grounded}, "
                f"confidence={result.groundedness.overall_confidence:.2f})"
            )
            for v in result.groundedness.claim_verdicts:
                print(
                    f"   - claim={v.claim.text!r} supported={v.supported} "
                    f"confidence={v.confidence:.2f} reason={v.reason}"
                )


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

    for query in QUERIES:
        print(f"\n{'=' * 80}\nQuery: {query!r}\n{'=' * 80}")
        result = answer(query, qdrant_client, openai_client, sparse_model, cohere_client)
        _print_result(result)


if __name__ == "__main__":
    main()
