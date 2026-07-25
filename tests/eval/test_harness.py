from libs.eval import harness as harness_module
from libs.eval.dataset import EvalCase
from libs.eval.harness import run_eval
from libs.generation.pipeline import AnswerResult
from libs.retrieval.pipeline import RetrievedResult

_CASES = [
    EvalCase(case_id="pass-case", query="q1", expected_answered=False),
    EvalCase(case_id="fail-case", query="q2", expected_answered=False),
]


def test_run_eval_scores_every_case_and_aggregates_pass_count(monkeypatch):
    def fake_retrieve(query_text, *a, **k):
        return [RetrievedResult(score=0.5, payload={"chunk_id": f"{query_text}:chunk"})]

    def fake_answer(query_text, *a, **k):
        # q1 correctly refuses (matches expected); q2 incorrectly answers.
        if query_text == "q1":
            return AnswerResult(query=query_text, answered=False, refusal_reason="n/a")
        return AnswerResult(query=query_text, answered=True, answer="a guess", citations=[])

    monkeypatch.setattr(harness_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(harness_module, "answer", fake_answer)

    report = run_eval(_CASES, None, None, None, None)

    assert report.total == 2
    assert report.passed == 1
    assert report.pass_rate == 0.5
    results_by_id = {r.case_id: r for r in report.case_results}
    assert results_by_id["pass-case"].passed is True
    assert results_by_id["fail-case"].passed is False


def test_run_eval_with_no_cases_reports_zero_total(monkeypatch):
    monkeypatch.setattr(harness_module, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(harness_module, "answer", lambda *a, **k: None)

    report = run_eval([], None, None, None, None)

    assert report.total == 0
    assert report.passed == 0
    assert report.pass_rate == 0.0
