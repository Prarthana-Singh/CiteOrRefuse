"""Phase 5 investigation: isolates which stage (retrieval, generation, or
the groundedness gate) is responsible for the Tesla non-determinism found
during eval-harness hardening -- the same query against the same index
flipped between "answered confidently" and "refused" across runs.

Runs the exact same query 20 times, logging every intermediate stage (not
just the final verdict): post-rerank chunk_ids + scores, the raw generated
claims, and the groundedness gate's actual confidence value per run. Paces
calls explicitly (not just relying on reactive retry) to stay under a
Cohere trial key's 10-calls/minute cap across 20 sequential runs.

Requires real OPENAI_API_KEY and COHERE_API_KEY.

Usage: python scripts/investigate_tesla_flakiness.py
"""
import json
import time
from pathlib import Path

import cohere
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

from libs.chunking.pipeline import chunk_filing_from_source
from libs.core.config import groundedness_settings, vectorstore_settings
from libs.core.models import Filing
from libs.generation.context_builder import build_context
from libs.generation.generator import generate_answer
from libs.generation.groundedness import check_groundedness
from libs.indexing.pipeline import index_filing
from libs.retrieval.pipeline import retrieve

QUERY = "What are Tesla's reportable business segments?"
NUM_RUNS = 20
PACING_SECONDS = 6.5  # stay under Cohere trial's 10 calls/minute


def main() -> None:
    filing = Filing(
        filing_id="tsla-2025-10k",
        company="Tesla, Inc.",
        cik="0001318605",
        fiscal_year=2025,
        form_type="10-K",
        source_path="data/fixtures/tsla-20251231.html",
    )
    print(f"Ingesting + chunking + indexing tsla-20251231.html...")
    chunks = chunk_filing_from_source(filing)

    openai_client = OpenAI()
    qdrant_client = QdrantClient(":memory:")
    sparse_model = SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)
    cohere_client = cohere.ClientV2()

    index_filing(chunks, filing, openai_client, qdrant_client, sparse_model)

    runs = []
    for i in range(1, NUM_RUNS + 1):
        t0 = time.time()

        retrieved = retrieve(QUERY, qdrant_client, openai_client, sparse_model, cohere_client)
        retrieval_snapshot = [
            {"chunk_id": r.payload["chunk_id"], "score": round(r.score, 6)} for r in retrieved
        ]

        context = build_context(retrieved)
        generated = generate_answer(QUERY, context, openai_client)
        claims_snapshot = [{"text": c.text, "chunk_id": c.chunk_id} for c in generated.claims]

        groundedness = check_groundedness(generated, retrieved, openai_client)
        verdicts_snapshot = [
            {
                "claim_text": v.claim.text,
                "chunk_id": v.claim.chunk_id,
                "supported": v.supported,
                "confidence": round(v.confidence, 6),
                "reason": v.reason,
            }
            for v in groundedness.claim_verdicts
        ]

        would_answer = (
            groundedness.is_grounded
            and groundedness.overall_confidence >= groundedness_settings.confidence_threshold
        )

        run_record = {
            "run": i,
            "retrieved_top5": retrieval_snapshot,
            "generated_answer": generated.answer,
            "claims": claims_snapshot,
            "is_grounded": groundedness.is_grounded,
            "overall_confidence": round(groundedness.overall_confidence, 6),
            "claim_verdicts": verdicts_snapshot,
            "would_answer": would_answer,
        }
        runs.append(run_record)

        print(
            f"[run {i:2d}] would_answer={would_answer!s:5s} "
            f"is_grounded={groundedness.is_grounded!s:5s} "
            f"confidence={groundedness.overall_confidence:.4f} "
            f"claims={len(generated.claims)} "
            f"top_chunk={retrieval_snapshot[0]['chunk_id'] if retrieval_snapshot else None}"
        )

        elapsed = time.time() - t0
        if i < NUM_RUNS and elapsed < PACING_SECONDS:
            time.sleep(PACING_SECONDS - elapsed)

    out_path = Path(__file__).parent.parent / "data" / "eval" / "_tesla_flakiness_runs.json"
    out_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\nWrote {len(runs)} run records to {out_path}")

    answered_count = sum(1 for r in runs if r["would_answer"])
    print(f"\n{answered_count}/{NUM_RUNS} would answer, {NUM_RUNS - answered_count}/{NUM_RUNS} would refuse")


if __name__ == "__main__":
    main()
