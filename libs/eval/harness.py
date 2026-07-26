"""Runs the full answer() pipeline for every EvalCase and scores it -- the
CI-gated (manually-triggered, see .github/workflows/eval.yml) regression
check for retrieval/generation quality.
"""
import time

import cohere
from fastembed import SparseTextEmbedding
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient

from libs.core.logging import get_logger
from libs.eval.dataset import EvalCase
from libs.eval.metrics import CaseResult, score_case
from libs.generation.pipeline import answer

logger = get_logger(__name__)

# Cohere's trial key caps rerank calls at 10/minute. Each eval case now makes
# exactly one rerank call (via answer() -> retrieve()), so pacing between
# cases at this interval keeps the harness proactively under that cap
# instead of relying on reactive retry-with-backoff inside reranker.py to
# recover after the fact. Same reasoning/value as
# scripts/investigate_tesla_flakiness.py's pacing.
_PACING_SECONDS = 6.5


class EvalReport(BaseModel):
    case_results: list[CaseResult]
    total: int
    passed: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def run_eval(
    cases: list[EvalCase],
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
    sparse_model: SparseTextEmbedding,
    cohere_client: cohere.ClientV2,
) -> EvalReport:
    """Runs `answer()` once per case.

    Retrieval recall needs the raw top-K retrieved chunk_ids, which
    `AnswerResult` now exposes directly via `retrieved_chunk_ids` -- this
    used to call `retrieve()` a second time here to get that list, which
    silently doubled the Cohere rerank calls this harness made per case (one
    inside this loop, one again inside `answer()`) and was the actual cause
    of trial-tier rate-limit failures partway through a full eval run, not
    just a cosmetic double-logged line.
    """
    case_results: list[CaseResult] = []
    for i, case in enumerate(cases):
        t0 = time.time()

        result = answer(case.query, qdrant_client, openai_client, sparse_model, cohere_client)
        case_results.append(score_case(case, result, result.retrieved_chunk_ids))

        elapsed = time.time() - t0
        if i < len(cases) - 1 and elapsed < _PACING_SECONDS:
            time.sleep(_PACING_SECONDS - elapsed)

    passed = sum(1 for r in case_results if r.passed)
    logger.info("Eval run: %d/%d cases passed", passed, len(cases))
    return EvalReport(case_results=case_results, total=len(cases), passed=passed)
