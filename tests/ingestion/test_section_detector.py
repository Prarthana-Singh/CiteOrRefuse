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


def test_raises_when_no_items_found():
    blocks = [_text_block(0, "Just some prose with no item headings at all.")]
    with pytest.raises(SectionDetectionError):
        detect_sections(blocks)
