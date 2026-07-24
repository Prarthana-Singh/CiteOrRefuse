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


def test_single_row_layout_table_becomes_text_block():
    """Real EDGAR filings lay out headings in a one-row <table> for column
    alignment (e.g. "Item 1." / "Business" as two <td>s) rather than a
    semantic tag; such a table has no body rows and must surface as text
    so section detection can see it, instead of being extracted as data.
    """
    html = "<html><body><table><tr><td>Item&nbsp;1.</td><td>Business</td></tr></table></body></html>"
    blocks = SecHtmlParser().parse(html).blocks
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.TEXT
    assert blocks[0].text == "Item 1. Business"


def test_parses_non_utf8_bytes_using_declared_charset():
    """Real EDGAR filings are routinely saved in windows-1252, not UTF-8.

    Passing raw bytes (rather than a pre-decoded str) lets lxml sniff the
    declared <meta charset> itself; pre-decoding in Python and handing lxml
    a large str full of non-Latin-1 characters has been observed to
    silently truncate the parse tree with no exception raised at all.
    """
    html = (
        "<html><head><meta charset=\"windows-1252\"></head>"
        "<body><p>registrant’s report</p></body></html>"
    )
    raw_bytes = html.encode("windows-1252")
    blocks = SecHtmlParser().parse(raw_bytes).blocks
    assert len(blocks) == 1
    assert blocks[0].text == "registrant’s report"


def test_word_split_across_adjacent_inline_tags_is_not_space_joined():
    """Real EDGAR filings (Microsoft's FY2025 10-K, at least) routinely split
    a single word across two adjacent inline tags with no whitespace between
    them in the source -- e.g. "<span>STATE</span><span>MENTS</span>" for
    "STATEMENTS", apparently a pagination/reflow artifact. bs4's
    get_text(separator=" ") used to insert a space at every such boundary
    regardless, corrupting the word into "STATE MENTS".
    """
    html = (
        "<html><body><p>"
        "<span>ITEM 8. FINANCIAL STATE</span><span>MENTS AND SUPPLEMENTARY DATA</span>"
        "</p></body></html>"
    )
    blocks = SecHtmlParser().parse(html).blocks
    assert blocks[0].text == "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA"


def test_br_tag_still_separates_text_with_a_space():
    """A <br> contributes no text of its own, so removing the artificial
    separator (to fix mid-word splitting, above) must not let text on
    either side of a line break fuse together.
    """
    html = "<html><body><p>Line one<br/>Line two</p></body></html>"
    blocks = SecHtmlParser().parse(html).blocks
    assert blocks[0].text == "Line one Line two"


def test_multi_row_table_is_still_extracted_as_table():
    html = (
        "<html><body><table>"
        "<tr><td>Metric</td><td>FY23</td></tr>"
        "<tr><td>Revenue</td><td>$100</td></tr>"
        "</table></body></html>"
    )
    blocks = SecHtmlParser().parse(html).blocks
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.TABLE
