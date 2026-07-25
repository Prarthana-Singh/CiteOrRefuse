from libs.eval.dataset import EvalCase
from libs.eval.metrics import score_case
from libs.generation.generator import Claim
from libs.generation.groundedness import ClaimVerdict, GroundednessResult
from libs.generation.pipeline import AnswerResult

_REFUSAL_CASE = EvalCase(
    case_id="refusal", query="What color is the sky on Mars?", expected_answered=False
)

_ANSWER_CASE = EvalCase(
    case_id="answer",
    query="What were net sales?",
    expected_answered=True,
    expected_chunk_ids=["acme-10k:7:0"],
    expected_answer_contains=["$1B"],
)


def _refused_result(query: str) -> AnswerResult:
    return AnswerResult(query=query, answered=False, refusal_reason="insufficient evidence")


def _answered_result(
    query: str, answer_text: str, cited_chunk_ids: list[str]
) -> AnswerResult:
    return AnswerResult(
        query=query,
        answered=True,
        answer=answer_text,
        citations=[{"chunk_id": cid} for cid in cited_chunk_ids],
    )


def test_correct_refusal_passes():
    result = _refused_result(_REFUSAL_CASE.query)

    scored = score_case(_REFUSAL_CASE, result, retrieved_chunk_ids=[])

    assert scored.passed is True
    assert scored.refusal_correct is True


def test_answering_when_a_refusal_was_expected_fails():
    result = _answered_result(_REFUSAL_CASE.query, "The sky is red.", ["some:chunk:0"])

    scored = score_case(_REFUSAL_CASE, result, retrieved_chunk_ids=["some:chunk:0"])

    assert scored.passed is False
    assert scored.refusal_correct is False
    assert "expected answered=False" in scored.details


def test_correct_answer_with_expected_chunk_and_content_passes():
    result = _answered_result(_ANSWER_CASE.query, "Net sales were $1B.", ["acme-10k:7:0"])

    scored = score_case(
        _ANSWER_CASE, result, retrieved_chunk_ids=["acme-10k:7:0", "acme-10k:7:1"]
    )

    assert scored.passed is True
    assert scored.retrieval_recall == 1.0
    assert scored.citation_correct is True
    assert scored.content_correct is True


def test_refusing_when_an_answer_was_expected_fails():
    result = _refused_result(_ANSWER_CASE.query)

    scored = score_case(_ANSWER_CASE, result, retrieved_chunk_ids=["acme-10k:7:0"])

    assert scored.passed is False
    assert scored.refusal_correct is False


def test_retrieval_missing_the_expected_chunk_fails_even_if_the_final_answer_is_correct():
    """Retrieval recall is checked against the RAW retrieved set, not just
    whether the final answer happened to look right -- a case could pass
    on citation/content by luck while retrieval itself regressed.
    """
    result = _answered_result(_ANSWER_CASE.query, "Net sales were $1B.", ["acme-10k:7:0"])

    scored = score_case(_ANSWER_CASE, result, retrieved_chunk_ids=["some:other:chunk"])

    assert scored.passed is False
    assert scored.retrieval_recall == 0.0


def test_answer_citing_the_wrong_chunk_fails_citation_check():
    result = _answered_result(_ANSWER_CASE.query, "Net sales were $1B.", ["wrong:chunk:id"])

    scored = score_case(
        _ANSWER_CASE, result, retrieved_chunk_ids=["acme-10k:7:0", "wrong:chunk:id"]
    )

    assert scored.passed is False
    assert scored.citation_correct is False


def test_answer_missing_expected_content_fails_content_check():
    result = _answered_result(_ANSWER_CASE.query, "Net sales grew significantly.", ["acme-10k:7:0"])

    scored = score_case(_ANSWER_CASE, result, retrieved_chunk_ids=["acme-10k:7:0"])

    assert scored.passed is False
    assert scored.content_correct is False


_ZERO_CLAIMS_CASE = EvalCase(
    case_id="zero-claims",
    query="What was Amazon's North America net sales in 2026?",
    expected_answered=False,
    assert_zero_claims=True,
)


def test_assert_zero_claims_passes_when_generation_produced_no_claims():
    result = AnswerResult(
        query=_ZERO_CLAIMS_CASE.query,
        answered=False,
        refusal_reason="insufficient evidence",
        groundedness=GroundednessResult(is_grounded=False, overall_confidence=0.0, claim_verdicts=[]),
    )

    scored = score_case(_ZERO_CLAIMS_CASE, result, retrieved_chunk_ids=["amzn:8:table:157"])

    assert scored.passed is True
    assert scored.zero_claims_correct is True


def test_assert_zero_claims_passes_when_retrieval_found_nothing_at_all():
    """No candidates retrieved -> generation never ran -> groundedness is
    None -- this trivially satisfies "zero claims" too, not a violation."""
    result = AnswerResult(query=_ZERO_CLAIMS_CASE.query, answered=False, groundedness=None)

    scored = score_case(_ZERO_CLAIMS_CASE, result, retrieved_chunk_ids=[])

    assert scored.passed is True


def test_assert_zero_claims_fails_when_the_gate_caught_nonzero_claims():
    """Refused correctly, but via the groundedness gate rejecting generated
    claims -- not via generation producing zero claims in the first place.
    A real, distinct failure mode from "answered when it should refuse"."""
    claim = Claim(text="Revenue was $2B in 2026.", chunk_id="amzn:8:table:157")
    result = AnswerResult(
        query=_ZERO_CLAIMS_CASE.query,
        answered=False,
        refusal_reason="insufficient evidence",
        groundedness=GroundednessResult(
            is_grounded=False,
            overall_confidence=0.9,
            claim_verdicts=[ClaimVerdict(claim=claim, supported=False, confidence=0.9)],
        ),
    )

    scored = score_case(_ZERO_CLAIMS_CASE, result, retrieved_chunk_ids=["amzn:8:table:157"])

    assert scored.passed is False
    assert scored.zero_claims_correct is False
    assert "groundedness gate" in scored.details


def test_assert_zero_claims_fails_when_the_pipeline_answers_instead_of_refusing():
    """Regression scenario found live in Phase 5 Task 3b: the pipeline
    doesn't refuse at all, it fully answers -- the failure detail must say
    so, not claim a refusal happened via the gate when none did.
    """
    claim = Claim(text="Revenue was $2B in 2026.", chunk_id="amzn:8:table:157")
    result = AnswerResult(
        query=_ZERO_CLAIMS_CASE.query,
        answered=True,
        answer="Revenue was $2B in 2026.",
        citations=[{"chunk_id": "amzn:8:table:157"}],
        groundedness=GroundednessResult(
            is_grounded=True,
            overall_confidence=0.95,
            claim_verdicts=[ClaimVerdict(claim=claim, supported=True, confidence=0.95)],
        ),
    )

    scored = score_case(_ZERO_CLAIMS_CASE, result, retrieved_chunk_ids=["amzn:8:table:157"])

    assert scored.passed is False
    assert scored.zero_claims_correct is False
    assert "answered instead" in scored.details
    assert "groundedness gate" not in scored.details


def test_case_without_expected_chunk_ids_or_content_only_checks_the_refused_flag():
    loose_case = EvalCase(case_id="loose", query="q", expected_answered=True)
    result = _answered_result("q", "anything at all", ["any:chunk"])

    scored = score_case(loose_case, result, retrieved_chunk_ids=[])

    assert scored.passed is True
    assert scored.retrieval_recall is None
    assert scored.citation_correct is None
    assert scored.content_correct is None
