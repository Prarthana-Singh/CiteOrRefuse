from pathlib import Path

import pytest

from libs.core.exceptions import SectionDetectionError
from libs.ingestion.parsers.base import Block, BlockType
from libs.ingestion.parsers.html_parser import SecHtmlParser
from libs.ingestion.section_detector.sec_10k_sections import detect_sections

FIXTURE_PATH = Path(__file__).parents[2] / "data" / "fixtures" / "sample_10k_excerpt.htm"


def _text_block(index: int, text: str) -> Block:
    return Block(index=index, block_type=BlockType.TEXT, text=text)


def test_detects_all_items_in_fixture_with_correct_parts():
    raw_html = FIXTURE_PATH.read_text(encoding="utf-8")
    blocks = SecHtmlParser().parse(raw_html).blocks
    sections = detect_sections(blocks)

    codes = [s.item_code for s in sections]
    assert codes == ["1", "1A", "7", "8"]
    assert sections[0].title == "Business"
    assert sections[1].title == "Risk Factors"
    assert sections[0].part == "I"
    assert sections[1].part == "I"
    assert sections[2].part == "II"
    assert sections[3].part == "II"


def test_section_boundaries_do_not_overlap():
    raw_html = FIXTURE_PATH.read_text(encoding="utf-8")
    blocks = SecHtmlParser().parse(raw_html).blocks
    sections = detect_sections(blocks)
    for prev, nxt in zip(sections, sections[1:]):
        assert prev.end_block_idx < nxt.start_block_idx


def test_toc_dot_leader_entries_are_ignored():
    blocks = [
        _text_block(0, "Item 1. Business ......................... 4"),
        _text_block(1, "Item 1A. Risk Factors .................... 9"),
        _text_block(2, "Item 1. Business"),
        _text_block(3, "Real business content here."),
    ]
    sections = detect_sections(blocks)
    assert len(sections) == 1
    assert sections[0].start_block_idx == 2


def test_lookahead_fills_title_when_split_across_blocks():
    blocks = [
        _text_block(0, "Item 1A."),
        _text_block(1, "Risk Factors"),
        _text_block(2, "Body content."),
    ]
    sections = detect_sections(blocks)
    assert sections[0].title == "Risk Factors"


def test_repeated_running_header_collapses_to_one_section():
    """Real EDGAR filings paginate long sections by repeating a bare "Item
    N" running header at the top of every printed page. This used to
    fragment a single section into one per page, each with a junk title
    pulled from whatever boilerplate happened to sit near the header.
    """
    blocks = [
        _text_block(0, "PART II"),
        _text_block(1, "Item 8"),  # running header: no title on this line
        _text_block(2, "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA"),
        _text_block(3, "Some financial statement content."),
        _text_block(4, "PART II"),
        _text_block(5, "Item 8"),  # repeated running header, next printed page
        _text_block(6, "More financial statement content."),
        _text_block(7, "PART II"),
        _text_block(8, "Item 9"),
        _text_block(9, "ITEM 9. CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS"),
    ]
    sections = detect_sections(blocks)

    item8_sections = [s for s in sections if s.item_code == "8"]
    assert len(item8_sections) == 1
    assert item8_sections[0].title == "FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA"
    assert item8_sections[0].start_block_idx == 1


def test_transitional_breadcrumb_does_not_win_over_real_heading():
    """Some filings print a transitional breadcrumb like "Item 1B, 1C" at
    page boundaries. It parses as a *direct* match (title captured on the
    same line as the item number) same as a real heading does, but with a
    garbage title (", 1C") -- a real heading's title starts with a letter,
    a breadcrumb's doesn't, which is what should break the tie.
    """
    blocks = [
        _text_block(0, "Item 1B, 1C"),
        _text_block(1, "ITEM 1B. UNRESOLVED STAFF COMMENTS"),
        _text_block(2, "Body content."),
    ]
    sections = detect_sections(blocks)

    assert sections[0].item_code == "1B"
    assert sections[0].title == "UNRESOLVED STAFF COMMENTS"


def test_part_header_with_trailing_description_is_recognized():
    """Not every filer writes a bare "PART I" on its own line -- Apple's
    10-Q writes "PART I – FINANCIAL INFORMATION", which the old
    end-anchored `_PART_RE` (requiring nothing after the roman numeral)
    silently failed to match, leaving every section's `part` as None.
    """
    blocks = [
        _text_block(0, "PART I – FINANCIAL INFORMATION"),
        _text_block(1, "Item 1. Financial Statements"),
        _text_block(2, "Body content."),
        _text_block(3, "PART II – OTHER INFORMATION"),
        _text_block(4, "Item 1. Legal Proceedings"),
        _text_block(5, "Body content."),
    ]
    sections = detect_sections(blocks)
    assert [s.part for s in sections] == ["I", "II"]


def test_toc_dot_leader_part_entry_is_ignored():
    """A TOC line like "Part II – Other Information ..... 45" must not be
    treated as a real Part header (which would prematurely flip
    `current_part` before the real Part I content it's still listing).
    """
    blocks = [
        _text_block(0, "Part I – Financial Information ......... 3"),
        _text_block(1, "Part II – Other Information ......... 45"),
        _text_block(2, "PART I – FINANCIAL INFORMATION"),
        _text_block(3, "Item 1. Financial Statements"),
        _text_block(4, "Body content."),
    ]
    sections = detect_sections(blocks)
    assert sections[0].part == "I"


def test_raises_when_no_items_found():
    blocks = [_text_block(0, "Just some prose with no item headings at all.")]
    with pytest.raises(SectionDetectionError):
        detect_sections(blocks)
