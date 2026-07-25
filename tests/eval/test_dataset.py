import json

from libs.eval.dataset import EvalCase, load_eval_cases

_SAMPLE_CASES = [
    {
        "case_id": "case-1",
        "query": "What were sales in 2025?",
        "expected_answered": True,
        "expected_chunk_ids": ["acme-10k:7:0"],
        "expected_answer_contains": ["$1B"],
    },
    {
        "case_id": "case-2",
        "query": "What color is the sky on Mars?",
        "expected_answered": False,
    },
]


def test_load_eval_cases_parses_all_fields(tmp_path):
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(_SAMPLE_CASES), encoding="utf-8")

    cases = load_eval_cases(path)

    assert len(cases) == 2
    assert cases[0] == EvalCase(
        case_id="case-1",
        query="What were sales in 2025?",
        expected_answered=True,
        expected_chunk_ids=["acme-10k:7:0"],
        expected_answer_contains=["$1B"],
    )


def test_load_eval_cases_defaults_optional_fields_to_empty(tmp_path):
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps([_SAMPLE_CASES[1]]), encoding="utf-8")

    cases = load_eval_cases(path)

    assert cases[0].expected_chunk_ids == []
    assert cases[0].expected_answer_contains == []
    assert cases[0].assert_zero_claims is False
    assert cases[0].notes is None


def test_real_seed_eval_set_loads_and_parses():
    """The actual checked-in seed dataset must always be valid."""
    from pathlib import Path

    seed_path = Path(__file__).parents[2] / "data" / "eval" / "sec_filings_eval_set.json"

    cases = load_eval_cases(seed_path)

    assert len(cases) == 17
    assert all(isinstance(c, EvalCase) for c in cases)
    refusal_cases = [c for c in cases if not c.expected_answered]
    assert len(refusal_cases) == 9
    # Every fixture (Amazon, Tesla, Microsoft, Apple) must have real cases.
    case_ids = {c.case_id for c in cases}
    assert any(cid.startswith("tesla-") for cid in case_ids)
    assert any(cid.startswith("msft-") for cid in case_ids)
    assert any(cid.startswith("apple-") for cid in case_ids)
