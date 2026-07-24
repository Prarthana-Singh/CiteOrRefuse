from pathlib import Path

import pytest

from libs.ingestion.parsers.base import BlockType
from libs.ingestion.parsers.html_parser import SecHtmlParser

FIXTURE_PATH = Path(__file__).parents[2] / "data" / "fixtures" / "sample_10k_excerpt.htm"


@pytest.fixture(scope="module")
def parsed_blocks():
    raw_html = FIXTURE_PATH.read_text(encoding="utf-8")
    return SecHtmlParser().parse(raw_html).blocks


def test_boilerplate_is_filtered_out(parsed_blocks):
    texts = [b.text for b in parsed_blocks if b.block_type == BlockType.TEXT]
    assert not any(t.strip().lower() == "table of contents" for t in texts)
    assert not any(t.strip() == "12" for t in texts)


def test_blocks_are_in_document_order(parsed_blocks):
    texts = [b.text for b in parsed_blocks if b.block_type == BlockType.TEXT]
    business_idx = next(i for i, t in enumerate(texts) if "Item 1. Business" in t)
    risk_idx = next(i for i, t in enumerate(texts) if "Item 1A" in t)
    mdna_idx = next(i for i, t in enumerate(texts) if "Item 7" in t)
    assert business_idx < risk_idx < mdna_idx


def test_style_based_headings_are_flagged_emphasized(parsed_blocks):
    heading = next(b for b in parsed_blocks if b.block_type == BlockType.TEXT and "Item 1A" in b.text)
    assert heading.is_emphasized is True


def test_table_block_is_extracted_with_structure(parsed_blocks):
    table_blocks = [b for b in parsed_blocks if b.block_type == BlockType.TABLE]
    assert len(table_blocks) == 1
    table = table_blocks[0].table
    assert table is not None
    assert table.caption == "Selected Financial Data (in thousands, except per share data)"
    assert "Total revenue" in table.rows[0]
    # The fully-empty spacer row between "Total revenue" and "Gross profit" must be dropped.
    assert all(any(cell for cell in row) for row in table.rows)


def test_block_indices_are_sequential(parsed_blocks):
    assert [b.index for b in parsed_blocks] == list(range(len(parsed_blocks)))
