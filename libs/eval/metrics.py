"""Scores a single pipeline run against its EvalCase's expectations."""
from pydantic import BaseModel

from libs.eval.dataset import EvalCase
from libs.generation.pipeline import AnswerResult


class CaseResult(BaseModel):
    case_id: str
    query: str
    passed: bool
    refusal_correct: bool
    zero_claims_correct: bool = True
    retrieval_recall: float | None = None
    citation_correct: bool | None = None
    content_correct: bool | None = None
    details: str = ""


def score_case(case: EvalCase, result: AnswerResult, retrieved_chunk_ids: list[str]) -> CaseResult:
    """Scores one case.

    `retrieved_chunk_ids` are the chunk_ids actually retrieved (top-K,
    before generation/groundedness ever ran) for this case's query --
    needed to measure retrieval recall independent of whether the pipeline
    ultimately answered or refused, since a correct refusal on a
    topically-on-target-but-unanswerable query (see Phase 4's stress test)
    can coexist with perfectly fine retrieval.
    """
    refusal_correct = result.answered == case.expected_answered

    zero_claims_correct = True
    if case.assert_zero_claims:
        # `AnswerResult` doesn't carry a claim count directly; `groundedness`
        # is only None when retrieval found zero candidates (generation
        # never ran at all, which trivially means zero claims too), and
        # `check_groundedness` builds exactly one ClaimVerdict per claim, so
        # its length is claim count whenever generation did run.
        claims_count = len(result.groundedness.claim_verdicts) if result.groundedness else 0
        zero_claims_correct = claims_count == 0

    retrieval_recall = None
    if case.expected_chunk_ids:
        hit = any(chunk_id in retrieved_chunk_ids for chunk_id in case.expected_chunk_ids)
        retrieval_recall = 1.0 if hit else 0.0

    citation_correct = None
    content_correct = None
    if case.expected_answered:
        if case.expected_chunk_ids and result.citations is not None:
            actual_ids = {c["chunk_id"] for c in result.citations}
            citation_correct = bool(actual_ids & set(case.expected_chunk_ids))
        if case.expected_answer_contains and result.answer is not None:
            answer_lower = result.answer.lower()
            content_correct = all(s.lower() in answer_lower for s in case.expected_answer_contains)

    passed = (
        refusal_correct
        and zero_claims_correct
        and (retrieval_recall is None or retrieval_recall == 1.0)
        and (citation_correct is None or citation_correct)
        and (content_correct is None or content_correct)
    )

    details = []
    if not refusal_correct:
        details.append(f"expected answered={case.expected_answered}, got {result.answered}")
    if not zero_claims_correct:
        details.append(
            f"expected zero claims from generation, got {claims_count} "
            f"(refusal happened via the groundedness gate, not generation itself)"
        )
    if retrieval_recall == 0.0:
        details.append(f"none of expected chunk_ids {case.expected_chunk_ids} were retrieved")
    if citation_correct is False:
        details.append("actual citations didn't include any expected chunk_id")
    if content_correct is False:
        details.append(f"answer text missing expected substring(s) {case.expected_answer_contains}")

    return CaseResult(
        case_id=case.case_id,
        query=case.query,
        passed=passed,
        refusal_correct=refusal_correct,
        zero_claims_correct=zero_claims_correct,
        retrieval_recall=retrieval_recall,
        citation_correct=citation_correct,
        content_correct=content_correct,
        details="; ".join(details),
    )
