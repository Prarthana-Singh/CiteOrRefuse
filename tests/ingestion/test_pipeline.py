from pathlib import Path

import pytest

from libs.core.exceptions import UnsupportedDocumentTypeError
from libs.core.models import Filing
from libs.ingestion.pipeline import ingest_filing

_FIXTURES_DIR = Path(__file__).parents[2] / "data" / "fixtures"

# Every real 10-K fixture below is expected to resolve to the same 23-item,
# 4-part canonical SEC 10-K structure -- Amazon, Tesla, and Microsoft's
# FY2025 filings all use it, despite very different HTML generators.
_EXPECTED_ITEM_CODES = [
    "1", "1A", "1B", "1C", "2", "3", "4",
    "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14",
    "15", "16",
]
_EXPECTED_PARTS = ["I"] * 7 + ["II"] * 9 + ["III"] * 5 + ["IV"] * 2


@pytest.mark.parametrize(
    "fixture_name,expected_block_count",
    [
        ("tsla-20251231.html", 988),
        ("10-K.html", 1173),  # Microsoft FY2025 10-K (msft-20250630.htm)
    ],
)
def test_ingest_real_filing_detects_all_23_items_in_order(fixture_name, expected_block_count):
    """Pins block/section counts against real EDGAR filings from two
    different HTML generators, so a change to the parser or section
    detector that silently regresses either is caught immediately rather
    than discovered the next time someone happens to eyeball the output.
    """
    filing = Filing(
        filing_id=fixture_name,
        company="Test Co",
        source_path=str(_FIXTURES_DIR / fixture_name),
    )

    blocks, sections = ingest_filing(filing)

    assert len(blocks) == expected_block_count
    assert [s.item_code for s in sections] == _EXPECTED_ITEM_CODES
    assert [s.part for s in sections] == _EXPECTED_PARTS


@pytest.mark.parametrize("fixture_name", ["tsla-20251231.html", "10-K.html"])
def test_ingest_real_filing_item_8_title_is_not_fragmented_by_pagination(fixture_name):
    """Regression guard for a Microsoft-filing-specific bug: long sections
    (financial statements worst of all) repeat a "PART II" / "Item 8"
    running header at the top of every printed page, which used to
    fragment Item 8 into ~40 near-empty sections instead of one, with junk
    titles pulled from whatever boilerplate happened to sit near a bare
    running header.
    """
    filing = Filing(
        filing_id=fixture_name,
        company="Test Co",
        source_path=str(_FIXTURES_DIR / fixture_name),
    )

    _, sections = ingest_filing(filing)

    item8_sections = [s for s in sections if s.item_code == "8"]
    assert len(item8_sections) == 1
    assert item8_sections[0].title.upper() == "FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA"


def test_ingest_real_10q_filing_uses_10q_item_structure():
    """`data/fixtures/aapl-20260328.html` is a 10-Q (quarterly report), not
    a 10-K, despite the fixture naming convention matching the other real
    filings -- confirmed via its `dei:DocumentType` tag. A 10-Q reuses Item
    codes 1-4 across both Part I and Part II with different meanings, and
    its Part headers ("PART I – FINANCIAL INFORMATION") carry a trailing
    description the other three filings' bare "PART I" headers don't,
    which is what exposed the `_PART_RE` bug this test guards against.
    """
    filing = Filing(
        filing_id="aapl-10q",
        company="Apple Inc.",
        form_type="10-Q",
        source_path=str(_FIXTURES_DIR / "aapl-20260328.html"),
    )

    blocks, sections = ingest_filing(filing)

    assert len(blocks) == 312
    assert [s.item_code for s in sections] == [
        "1", "2", "3", "4", "1", "1A", "2", "3", "4", "5", "6",
    ]
    assert [s.part for s in sections] == ["I"] * 4 + ["II"] * 7


def test_ingest_rejects_a_10q_filed_as_the_default_10k_form_type():
    """The exact accident the guard exists for: a Filing left at the
    default form_type="10-K" but actually pointed at a 10-Q. Section
    detection is tuned to the 10-K Item 1-16 structure and would otherwise
    silently produce a plausible-looking but wrong result, since a 10-Q
    reuses the same "Item N" heading pattern.
    """
    filing = Filing(
        filing_id="oops-wrong-type",
        company="Apple Inc.",
        source_path=str(_FIXTURES_DIR / "aapl-20260328.html"),
    )

    with pytest.raises(UnsupportedDocumentTypeError, match="10-Q"):
        ingest_filing(filing)


_DOCUMENT_TYPE_MISMATCH_HTML = (
    "<html><body>"
    '<ix:nonnumeric name="dei:DocumentType">10-Q</ix:nonnumeric>'
    "<div>Item 1. Financial Statements</div>"
    "<p>Some content.</p>"
    "</body></html>"
)


def test_ingest_rejects_declared_document_type_mismatch_synthetic(tmp_path):
    """Isolated, fast version of the guard check, independent of the large
    real-filing fixture: any declared `dei:DocumentType` that disagrees
    with `Filing.form_type` must raise before parsing proceeds.
    """
    source_path = tmp_path / "filing.htm"
    source_path.write_text(_DOCUMENT_TYPE_MISMATCH_HTML, encoding="utf-8")
    filing = Filing(filing_id="mismatch", company="Test Co", source_path=str(source_path))

    with pytest.raises(UnsupportedDocumentTypeError):
        ingest_filing(filing)


def test_ingest_accepts_declared_document_type_match_synthetic(tmp_path):
    """A `dei:DocumentType` that agrees with `Filing.form_type` (e.g. a
    10-Q correctly declared as such) must not be rejected.
    """
    source_path = tmp_path / "filing.htm"
    source_path.write_text(_DOCUMENT_TYPE_MISMATCH_HTML, encoding="utf-8")
    filing = Filing(
        filing_id="match", company="Test Co", form_type="10-Q", source_path=str(source_path)
    )

    blocks, sections = ingest_filing(filing)
    assert sections[0].item_code == "1"


_CP1252_FILING_HTML = (
    "<html><head><meta charset=\"windows-1252\"></head><body>"
    "<div>Item 1. Business</div>"
    "<p>The registrant’s principal business is described below.</p>"
    "<div>Item 1A. Risk Factors</div>"
    "<p>Various risks apply.</p>"
    "</body></html>"
)


def test_ingest_filing_reads_non_utf8_source_file(tmp_path):
    """Guards against a UnicodeDecodeError (or worse, a silent truncated
    parse) when a real EDGAR filing saved in windows-1252 is read from disk.
    """
    source_path = tmp_path / "filing.htm"
    source_path.write_bytes(_CP1252_FILING_HTML.encode("windows-1252"))

    filing = Filing(
        filing_id="test-10k",
        company="Test Co",
        source_path=str(source_path),
    )

    blocks, sections = ingest_filing(filing)

    assert [s.item_code for s in sections] == ["1", "1A"]
    business_block = next(b for b in blocks if "registrant" in b.text)
    assert "registrant’s" in business_block.text
