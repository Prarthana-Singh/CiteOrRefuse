"""Runs the eval harness against a shared index of all four real fixtures
(Amazon, Tesla, Microsoft 10-Ks; Apple 10-Q).

Requires real OPENAI_API_KEY and COHERE_API_KEY. Uses an in-memory Qdrant
instance, so nothing external needs to be running. Exits non-zero if any
case fails -- this is the actual CI gate (still manually-triggered, not
run on every PR, but a real pass/fail signal when it does run, not just a
report).

Usage: python scripts/run_eval.py [path/to/eval_set.json]
"""
import sys
from pathlib import Path

import cohere
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

from libs.core.config import vectorstore_settings
from libs.eval.dataset import load_eval_cases
from libs.eval.harness import run_eval
from libs.indexing.fixtures import FILINGS, index_all_fixtures

DEFAULT_EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "eval" / "sec_filings_eval_set.json"


def main() -> None:
    eval_set_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EVAL_SET_PATH

    openai_client = OpenAI()
    qdrant_client = QdrantClient(":memory:")
    sparse_model = SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)
    cohere_client = cohere.ClientV2()

    print(f"Ingesting + chunking + indexing {len(FILINGS)} fixture filings...")
    index_all_fixtures(openai_client, qdrant_client, sparse_model)

    cases = load_eval_cases(eval_set_path)
    print(f"\nRunning {len(cases)} eval cases from {eval_set_path.name}...\n")
    report = run_eval(cases, qdrant_client, openai_client, sparse_model, cohere_client)

    for r in report.case_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.case_id}: {r.query!r}")
        if not r.passed:
            print(f"       {r.details}")

    print(f"\n{report.passed}/{report.total} passed ({report.pass_rate:.0%})")

    if report.passed < report.total:
        sys.exit(1)


if __name__ == "__main__":
    main()
