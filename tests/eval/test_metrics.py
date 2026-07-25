from libs.eval.dataset import EvalCase
from libs.eval.metrics import score_case
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


def test_case_without_expected_chunk_ids_or_content_only_checks_the_refused_flag():
    loose_case = EvalCase(case_id="loose", query="q", expected_answered=True)
    result = _answered_result("q", "anything at all", ["any:chunk"])

    scored = score_case(loose_case, result, retrieved_chunk_ids=[])

    assert scored.passed is True
    assert scored.retrieval_recall is None
    assert scored.citation_correct is None
    assert scored.content_correct is None
