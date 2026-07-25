"""Phase 4 close-out: stress-tests the groundedness gate against queries
that are topically on-target (retrieval will surface real, plausible Item
1/7/8 content) but factually unanswerable from the filing -- the harder
failure mode between "clean match" and "clean non-match" that the earlier
smoke test (scripts/answer_amazon_fixture.py) didn't exercise.

For each query, prints: the top retrieved/reranked chunks, the generator's
raw structured output (answer + claims), and the full groundedness verdict
(per-claim supported/confidence/reason) -- not just the final answer/refuse
decision, so a false pass or an overly-conservative refusal is visible, not
just its outcome.

Requires real OPENAI_API_KEY and COHERE_API_KEY.

Usage: python scripts/stress_test_groundedness.py
"""
from pathlib import Path

import cohere
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

from libs.chunking.pipeline import chunk_filing_from_source
from libs.core.config import vectorstore_settings
from libs.core.models import Filing
from libs.generation.context_builder import build_context
from libs.generation.generator import generate_answer
from libs.generation.groundedness import check_groundedness
from libs.indexing.pipeline import index_filing
from libs.retrieval.pipeline import retrieve

FIXTURE_PATH = Path(__file__).parent.parent / "data" / "fixtures" / "amzn-20251231.html"

QUERIES = [
    "What was Amazon's North America net sales in 2026?",
    "What is Amazon's market share in the North America e-commerce market?",
    "How many Amazon Prime members are there worldwide as of the end of 2025?",
]


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
        print(f"\n{'=' * 90}\nQUERY: {query!r}\n{'=' * 90}")

        results = retrieve(query, qdrant_client, openai_client, sparse_model, cohere_client, top_k=5)
        print(f"\n--- Retrieval: top {len(results)} reranked candidates ---")
        for i, r in enumerate(results):
            p = r.payload
            preview = p["text"].replace("\n", " ")[:140]
            print(f"[{i}] score={r.score:.4f} Item {p['section_item_code']} ({p['section_title'][:45]})")
            print(f"     chunk_id={p['chunk_id']}  {preview}...")

        if not results:
            print("\n(no candidates retrieved -- would refuse before generation even runs)")
            continue

        context = build_context(results)
        generated = generate_answer(query, context, openai_client)
        print(f"\n--- Generation ---")
        print(f"answer: {generated.answer}")
        print(f"claims ({len(generated.claims)}):")
        for c in generated.claims:
            print(f"  - [{c.chunk_id}] {c.text}")

        groundedness = check_groundedness(generated, results, openai_client)
        print(f"\n--- Groundedness gate ---")
        print(f"is_grounded={groundedness.is_grounded}  overall_confidence={groundedness.overall_confidence:.2f}")
        for v in groundedness.claim_verdicts:
            print(
                f"  - claim={v.claim.text!r}\n"
                f"    chunk_id={v.claim.chunk_id}  supported={v.supported}  "
                f"confidence={v.confidence:.2f}  reason={v.reason}"
            )

        print(f"\n--- Final verdict ---")
        if groundedness.is_grounded and groundedness.overall_confidence >= 0.7:
            print(f"WOULD ANSWER: {generated.answer}")
        else:
            print("WOULD REFUSE")


if __name__ == "__main__":
    main()
