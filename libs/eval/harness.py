"""Runs the full retrieve() + answer() pipeline for every EvalCase and
scores it -- the CI-gated (manually-triggered, see .github/workflows/eval.yml)
regression check for retrieval/generation quality.
"""
import cohere
from fastembed import SparseTextEmbedding
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient

from libs.core.logging import get_logger
from libs.eval.dataset import EvalCase
from libs.eval.metrics import CaseResult, score_case
from libs.generation.pipeline import answer
from libs.retrieval.pipeline import retrieve

logger = get_logger(__name__)


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
    """Runs `retrieve()` and `answer()` independently per case.

    Retrieval recall needs the raw top-K retrieved chunk_ids, which
    `AnswerResult` doesn't expose (it only carries final *citations* --
    the subset generation chose to cite, a different, narrower thing than
    what was retrieved). Calling `retrieve()` a second time per case
    duplicates that one API round-trip rather than changing Phase 3/4's
    already-tested return shapes just for the eval harness's convenience.
    """
    case_results: list[CaseResult] = []
    for case in cases:
        retrieved = retrieve(case.query, qdrant_client, openai_client, sparse_model, cohere_client)
        retrieved_chunk_ids = [r.payload["chunk_id"] for r in retrieved]

        result = answer(case.query, qdrant_client, openai_client, sparse_model, cohere_client)
        case_results.append(score_case(case, result, retrieved_chunk_ids))

    passed = sum(1 for r in case_results if r.passed)
    logger.info("Eval run: %d/%d cases passed", passed, len(cases))
    return EvalReport(case_results=case_results, total=len(cases), passed=passed)
