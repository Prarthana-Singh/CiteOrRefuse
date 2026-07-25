from libs.core.config import generation_settings
from libs.core.models import ChunkType
from libs.generation.context_builder import build_context
from libs.retrieval.pipeline import RetrievedResult


def _result(chunk_id: str, text: str, score: float = 0.9) -> RetrievedResult:
    return RetrievedResult(
        score=score,
        payload={
            "chunk_id": chunk_id,
            "filing_id": "acme-10k",
            "company": "Acme Robotics, Inc.",
            "form_type": "10-K",
            "filing_date": None,
            "section_item_code": "7",
            "section_title": "Management's Discussion and Analysis",
            "part": "II",
            "chunk_type": ChunkType.TEXT.value,
            "order_index": 0,
            "char_start": 0,
            "char_end": len(text),
            "token_count": len(text.split()),
            "text": text,
        },
    )


def test_build_context_tags_each_block_with_its_chunk_id():
    context = build_context([_result("acme-10k:7:0", "Revenue grew 12% year over year.")])

    assert "chunk_id: acme-10k:7:0" in context
    assert "Revenue grew 12% year over year." in context
    assert "Acme Robotics, Inc." in context


def test_build_context_dedupes_exact_duplicate_text():
    duplicate_text = "Revenue grew 12% year over year."
    results = [
        _result("acme-10k:7:0", duplicate_text),
        _result("acme-10k:8:table:0", duplicate_text),  # same text, different chunk
    ]

    context = build_context(results)

    assert context.count(duplicate_text) == 1


def test_build_context_preserves_distinct_overlapping_content():
    """Near-duplicate (not exact) content across sections -- e.g. the same
    figures in both an Item 7 table and an Item 8 table -- must NOT be
    deduped; only exact-text repeats are a dedup target.
    """
    results = [
        _result("acme-10k:7:table:0", "North America net sales: $426,305"),
        _result("acme-10k:8:table:0", "North America net sales (in millions): $426,305"),
    ]

    context = build_context(results)

    assert "acme-10k:7:table:0" in context
    assert "acme-10k:8:table:0" in context


def test_build_context_always_includes_at_least_one_block(monkeypatch):
    """Even a single result that alone exceeds the token budget must not
    be dropped -- an empty context is a worse failure than a slightly
    over-budget one.
    """
    monkeypatch.setattr(generation_settings, "context_token_budget", 1)

    context = build_context([_result("acme-10k:7:0", "A fairly long piece of text here.")])

    assert "acme-10k:7:0" in context


def test_build_context_drops_lowest_relevance_results_once_over_budget(monkeypatch):
    monkeypatch.setattr(generation_settings, "context_token_budget", 20)

    results = [
        _result("acme-10k:7:0", "First result text that takes up a fair number of tokens."),
        _result("acme-10k:7:1", "Second result text, also fairly long, taking more tokens."),
        _result("acme-10k:7:2", "Third result text, again long enough to add real token weight."),
    ]

    context = build_context(results)

    assert "acme-10k:7:0" in context
    assert "acme-10k:7:2" not in context
