"""Detects SEC 10-K `Item` sections by matching text patterns on short blocks.

Real EDGAR filings routinely mark headings with bold/underline styling
instead of semantic <h1>-<h6> tags, so this intentionally matches on text
pattern across all short TEXT blocks rather than requiring a heading tag.
"""
import re

from libs.core.exceptions import SectionDetectionError
from libs.core.models import Section
from libs.ingestion.parsers.base import Block, BlockType

_MAX_HEADING_LENGTH = 250

_ITEM_RE = re.compile(
    r"^\s*ITEM\s+(\d{1,2}[A-C]?)\.?\s*[–—:.-]?\s*(.*)$",
    re.IGNORECASE,
)
_PART_RE = re.compile(r"^\s*PART\s+(I{1,3}V?|IV)\.?\s*$", re.IGNORECASE)
_TOC_DOT_LEADER_RE = re.compile(r"\.{3,}")


def detect_sections(blocks: list[Block]) -> list[Section]:
    """Builds an ordered list of Sections from a filing's parsed blocks."""
    current_part: str | None = None
    matches: list[tuple[int, str, str, str | None]] = []

    for block in blocks:
        if block.block_type != BlockType.TEXT or len(block.text) > _MAX_HEADING_LENGTH:
            continue

        part_match = _PART_RE.match(block.text)
        if part_match:
            current_part = part_match.group(1).upper()
            continue

        if _TOC_DOT_LEADER_RE.search(block.text):
            # A table-of-contents entry (e.g. "Item 1A. Risk Factors ..... 12"),
            # not the actual section heading.
            continue

        item_match = _ITEM_RE.match(block.text)
        if item_match:
            item_code = item_match.group(1).upper()
            title = item_match.group(2).strip()
            if not title:
                title = _lookahead_title(blocks, block.index)
            matches.append((block.index, item_code, title, current_part))

    if not matches:
        raise SectionDetectionError("No SEC 10-K 'Item' sections were detected in this filing")

    sections: list[Section] = []
    for i, (start_idx, item_code, title, part) in enumerate(matches):
        end_idx = matches[i + 1][0] - 1 if i + 1 < len(matches) else blocks[-1].index
        sections.append(
            Section(
                item_code=item_code,
                title=title or f"Item {item_code}",
                part=part,
                start_block_idx=start_idx,
                end_block_idx=end_idx,
            )
        )
    return sections


def _lookahead_title(blocks: list[Block], item_block_idx: int) -> str:
    """Handles filings where the item number and title sit in separate blocks."""
    for block in blocks[item_block_idx + 1 : item_block_idx + 3]:
        if block.block_type == BlockType.TEXT and 0 < len(block.text) <= _MAX_HEADING_LENGTH:
            return block.text
    return ""
