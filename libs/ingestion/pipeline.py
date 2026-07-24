"""Orchestrates parse -> section detection for a single filing."""
import re
from pathlib import Path

from libs.core.exceptions import UnsupportedDocumentTypeError
from libs.core.logging import get_logger
from libs.core.models import Filing, Section
from libs.ingestion.parsers.base import Block
from libs.ingestion.parsers.html_parser import SecHtmlParser
from libs.ingestion.section_detector.sec_10k_sections import detect_sections

logger = get_logger(__name__)

# Matches the inline-XBRL tag EDGAR filings use to declare their SEC form
# type, e.g. <ix:nonnumeric name="dei:DocumentType" ...>10-K</ix:nonnumeric>.
_DOCUMENT_TYPE_TAG_RE = re.compile(rb'name=["\']dei:DocumentType["\']')
# The value isn't always plain text right after the tag -- some filers wrap
# it in a nested <span> for styling -- so rather than parse the intervening
# markup, just look for a known SEC form-type token within a bounded window
# after the tag. Cheaper than a second full lxml parse of a multi-MB filing,
# and this is a guard, not a general XBRL reader.
_DOCUMENT_TYPE_VALUE_RE = re.compile(rb"(10-K/A|10-K|10-Q/A|10-Q|8-K/A|8-K)\b")
_DOCUMENT_TYPE_SEARCH_WINDOW = 500


def _extract_document_type(raw_bytes: bytes) -> str | None:
    """Reads the filing's declared `dei:DocumentType`, or None if absent
    (e.g. a synthetic test fixture with no XBRL tagging at all)."""
    tag_match = _DOCUMENT_TYPE_TAG_RE.search(raw_bytes)
    if tag_match is None:
        return None
    window = raw_bytes[tag_match.end() : tag_match.end() + _DOCUMENT_TYPE_SEARCH_WINDOW]
    value_match = _DOCUMENT_TYPE_VALUE_RE.search(window)
    return value_match.group(1).decode("ascii") if value_match else None


def ingest_filing(filing: Filing) -> tuple[list[Block], list[Section]]:
    """Parses a filing's raw HTML and detects its SEC 10-K section structure."""
    # Raw bytes are handed to the parser rather than pre-decoded here: real
    # EDGAR filings are routinely saved (e.g. via a browser "Save As") in
    # whatever encoding their <meta charset> declares (often windows-1252,
    # not UTF-8), and lxml's own byte-level charset sniffing handles that
    # correctly end-to-end. Pre-decoding to a Python str first and handing
    # lxml a ~2MB string containing many non-Latin-1 characters (curly
    # quotes, em-dashes) has been observed to silently truncate the parse
    # tree partway through the document with no exception raised at all --
    # worse than a decode error, since the failure is invisible.
    raw_bytes = Path(filing.source_path).read_bytes()

    # Section detection is tuned to the 10-K Item 1-16 / Part I-IV structure
    # and will happily produce a plausible-looking but wrong result for a
    # 10-Q or 8-K, since those reuse the same "Item N" heading pattern. Fail
    # loudly here, before any parsing work happens, rather than let a
    # mismatched document type silently produce citations anchored to a
    # section structure the detector was never designed for.
    document_type = _extract_document_type(raw_bytes)
    if document_type is not None and document_type.upper() != filing.form_type.upper():
        raise UnsupportedDocumentTypeError(
            f"Filing {filing.filing_id!r} declares form_type={filing.form_type!r} "
            f"but its content is a {document_type} ({filing.source_path})"
        )

    parsed = SecHtmlParser().parse(raw_bytes)
    sections = detect_sections(parsed.blocks)
    logger.info(
        "Ingested filing %s: %d blocks, %d sections",
        filing.filing_id,
        len(parsed.blocks),
        len(sections),
    )
    return parsed.blocks, sections
