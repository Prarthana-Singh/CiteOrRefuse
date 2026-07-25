"""Eval dataset schema and loading."""
import json
from pathlib import Path

from pydantic import BaseModel


class EvalCase(BaseModel):
    """One eval case: a query plus what a correct answer should look like.

    A refusal case (`expected_answered=False`) doesn't need
    `expected_chunk_ids` or `expected_answer_contains` -- there's no answer
    to check citations or content against, only whether the pipeline
    correctly declined to guess.

    An answer case (`expected_answered=True`) should specify at least one
    of `expected_chunk_ids` / `expected_answer_contains` -- a case with
    neither can never actually fail on content, only on the
    answered/refused boolean.

    `assert_zero_claims` (refusal cases only) asserts the *mechanism* of
    the refusal, not just its outcome: that generation itself produced zero
    claims, rather than producing claims that the groundedness gate then
    caught (a hallucinated citation, or a judge verdict below threshold).
    Both are valid ways to end up refusing, but they're different pipeline
    behaviors -- a case can set this when the query is clearly enough
    outside the filing that generation should recognize it has nothing to
    cite, as opposed to a subtler case where a plausible-looking claim gets
    generated and only the gate catches it.
    """

    case_id: str
    query: str
    expected_answered: bool
    expected_chunk_ids: list[str] = []
    expected_answer_contains: list[str] = []
    assert_zero_claims: bool = False
    notes: str | None = None


def load_eval_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(**case) for case in data]
