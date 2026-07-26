from libs.eval import harness as harness_module
from libs.eval.dataset import EvalCase
from libs.eval.harness import run_eval
from libs.generation import pipeline as pipeline_module
from libs.generation.pipeline import AnswerResult

_CASES = [
    EvalCase(case_id="pass-case", query="q1", expected_answered=False),
    EvalCase(case_id="fail-case", query="q2", expected_answered=False),
]


def test_run_eval_scores_every_case_and_aggregates_pass_count(monkeypatch):
    def fake_answer(query_text, *a, **k):
        # q1 correctly refuses (matches expected); q2 incorrectly answers.
        if query_text == "q1":
            return AnswerResult(query=query_text, answered=False, refusal_reason="n/a")
        return AnswerResult(query=query_text, answered=True, answer="a guess", citations=[])

    monkeypatch.setattr(harness_module, "answer", fake_answer)
    monkeypatch.setattr(harness_module.time, "sleep", lambda seconds: None)

    report = run_eval(_CASES, None, None, None, None)

    assert report.total == 2
    assert report.passed == 1
    assert report.pass_rate == 0.5
    results_by_id = {r.case_id: r for r in report.case_results}
    assert results_by_id["pass-case"].passed is True
    assert results_by_id["fail-case"].passed is False


def test_run_eval_with_no_cases_reports_zero_total(monkeypatch):
    monkeypatch.setattr(harness_module, "answer", lambda *a, **k: None)
    monkeypatch.setattr(harness_module.time, "sleep", lambda seconds: None)

    report = run_eval([], None, None, None, None)

    assert report.total == 0
    assert report.passed == 0
    assert report.pass_rate == 0.0


def test_run_eval_calls_retrieve_exactly_once_per_case(monkeypatch):
    """Regression test for a real bug: `run_eval` used to call `retrieve()`
    directly to get retrieval-recall chunk_ids, *and* `answer()` called
    `retrieve()` again internally -- doubling the Cohere rerank calls made
    per eval case. That's what actually blew through the trial tier's
    10-calls/minute cap partway through a full run, not just a duplicated
    log line. `answer()` now exposes `retrieved_chunk_ids` on `AnswerResult`
    so `run_eval` never needs a `retrieve()` call of its own; this asserts
    the one `retrieve()` call `answer()` makes internally happens exactly
    once per case, not twice.
    """
    call_count = 0

    def counting_retrieve(*a, **k):
        nonlocal call_count
        call_count += 1
        return []

    monkeypatch.setattr(pipeline_module, "retrieve", counting_retrieve)
    monkeypatch.setattr(harness_module.time, "sleep", lambda seconds: None)

    run_eval(_CASES, None, None, None, None)

    assert call_count == len(_CASES)


def test_run_eval_paces_between_cases_but_not_after_the_last_one(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(harness_module, "answer", lambda *a, **k: AnswerResult(
        query="q", answered=False, refusal_reason="n/a"
    ))
    monkeypatch.setattr(harness_module.time, "sleep", sleep_calls.append)

    run_eval(_CASES, None, None, None, None)

    # 2 cases -> exactly one inter-case pause, none after the last case.
    assert len(sleep_calls) == 1
