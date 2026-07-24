"""Orchestrates parse -> section detection for a single filing."""
from pathlib import Path

from libs.core.logging import get_logger
from libs.core.models import Filing, Section
from libs.ingestion.parsers.base import Block
from libs.ingestion.parsers.html_parser import SecHtmlParser
from libs.ingestion.section_detector.sec_10k_sections import detect_sections

logger = get_logger(__name__)


def ingest_filing(filing: Filing) -> tuple[list[Block], list[Section]]:
    """Parses a filing's raw HTML and detects its SEC 10-K section structure."""
    raw_html = Path(filing.source_path).read_text(encoding="utf-8")
    parsed = SecHtmlParser().parse(raw_html)
    sections = detect_sections(parsed.blocks)
    logger.info(
        "Ingested filing %s: %d blocks, %d sections",
        filing.filing_id,
        len(parsed.blocks),
        len(sections),
    )
    return parsed.blocks, sections
