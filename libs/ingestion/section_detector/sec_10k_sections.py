"""Detects SEC 10-K `Item` sections by matching text patterns on short blocks.

Real EDGAR filings routinely mark headings with bold/underline styling
instead of semantic <h1>-<h6> tags, so this intentionally matches on text
pattern across all short TEXT blocks rather than requiring a heading tag.
"""
import re
from typing import NamedTuple

from libs.core.exceptions import SectionDetectionError
from libs.core.models import Section
from libs.ingestion.parsers.base import Block, BlockType

_MAX_HEADING_LENGTH = 250

_ITEM_RE = re.compile(
    r"^\s*ITEM\s+(\d{1,2}[A-C]?)\.?\s*[–—:.-]?\s*(.*)$",
    re.IGNORECASE,
)
_PART_RE = re.compile(r"^\s*PART\s+(I{1,3}V?|IV)\b", re.IGNORECASE)
_TOC_DOT_LEADER_RE = re.compile(r"\.{3,}")


def detect_sections(blocks: list[Block]) -> list[Section]:
    """Builds an ordered list of Sections from a filing's parsed blocks."""
    current_part: str | None = None
    matches: list[_Match] = []

    for block in blocks:
        if block.block_type != BlockType.TEXT or len(block.text) > _MAX_HEADING_LENGTH:
            continue

        if _TOC_DOT_LEADER_RE.search(block.text):
            # A table-of-contents entry (e.g. "Item 1A. Risk Factors ..... 12"
            # or "Part II – Other Information ..... 45"), not a real heading.
            continue

        part_match = _PART_RE.match(block.text)
        if part_match:
            current_part = part_match.group(1).upper()
            continue

        item_match = _ITEM_RE.match(block.text)
        if item_match:
            item_code = item_match.group(1).upper()
            title = item_match.group(2).strip()
            is_direct = bool(title)
            if not title:
                title = _lookahead_title(blocks, block.index)
            matches.append(_Match(block.index, item_code, title, current_part, is_direct))

    if not matches:
        raise SectionDetectionError("No SEC 10-K 'Item' sections were detected in this filing")

    matches = _dedupe_repeated_running_headers(matches)

    sections: list[Section] = []
    for i, match in enumerate(matches):
        end_idx = matches[i + 1].start_idx - 1 if i + 1 < len(matches) else blocks[-1].index
        sections.append(
            Section(
                item_code=match.item_code,
                title=match.title or f"Item {match.item_code}",
                part=match.part,
                start_block_idx=match.start_idx,
                end_block_idx=end_idx,
            )
        )
    return sections


class _Match(NamedTuple):
    start_idx: int
    item_code: str
    title: str
    part: str | None
    is_direct: bool


def _dedupe_repeated_running_headers(matches: list[_Match]) -> list[_Match]:
    """Collapses consecutive matches for the same (part, item code) into one.

    Grouping on item code alone would be wrong for document types (10-Qs)
    that reuse Item codes 1-4 across both Part I and Part II with different
    meanings -- if Part I's last item happened to share a code with Part
    II's first item and nothing separated them, they'd wrongly merge.
    Requiring the part to match too closes that gap; it's a no-op for the
    running-header case below since `current_part` never changes mid-run.

    Real EDGAR filings paginate long sections (financial statements are the
    worst offender) by repeating a "PART II" / "Item 8" running header at
    the top of every printed page. No 10-K legitimately repeats the same
    top-level Item as two separate sections, so once we are "inside" Item 8,
    further Item 8 headings are page-boundary noise, not new sections --
    collapsing them keeps a real section from being fragmented into dozens
    of near-empty ones.

    The run's *earliest* start index is kept (so the collapsed section still
    starts where the running header first appears, losing no content), but
    the *title* is taken from the best "direct" match in the run -- one
    where the title was captured on the same line as "Item N" itself --
    since bare running headers fall back to a 2-block lookahead that just as
    easily grabs unrelated boilerplate (a stray "PART II" line, a disclaimer
    paragraph) sitting near the header as it does the real title. Some
    filings also print a transitional breadcrumb like "Item 1B, 1C" at page
    boundaries, which itself parses as a *direct* match with a garbage
    title (", 1C"); a real heading's title starts with a letter, a
    breadcrumb's doesn't, so that's the tiebreaker among direct candidates.
    """
    runs: list[list[_Match]] = []
    for match in matches:
        if runs and (runs[-1][0].item_code, runs[-1][0].part) == (match.item_code, match.part):
            runs[-1].append(match)
        else:
            runs.append([match])

    deduped: list[_Match] = []
    for run in runs:
        direct_candidates = [m for m in run if m.is_direct]
        best = next((m for m in direct_candidates if m.title[:1].isalpha()), None)
        best = best or (direct_candidates[0] if direct_candidates else run[0])
        deduped.append(run[0]._replace(title=best.title))
    return deduped


def _lookahead_title(blocks: list[Block], item_block_idx: int) -> str:
    """Handles filings where the item number and title sit in separate blocks."""
    for block in blocks[item_block_idx + 1 : item_block_idx + 3]:
        if block.block_type != BlockType.TEXT or not (0 < len(block.text) <= _MAX_HEADING_LENGTH):
            continue
        if _PART_RE.match(block.text):
            # A running "PART II" header sitting between a bare "Item N"
            # running header and the real title is not a title candidate.
            continue
        return block.text
    return ""
