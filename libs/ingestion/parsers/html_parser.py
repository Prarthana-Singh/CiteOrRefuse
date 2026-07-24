"""Parses SEC EDGAR 10-K HTML into an ordered list of text/table blocks.

Real EDGAR filings routinely mark section headings with bold/underline
*styling* rather than semantic <h1>-<h6> tags (or any tag at all beyond a
plain <div>/<span>), so this parser does not try to detect headings itself.
It emits every leaf block of text in document order, tagging each with a
generic `is_emphasized` signal; the SEC-specific heading pattern matching
lives in `libs.ingestion.section_detector`, decoupled from HTML structure.
"""
from collections.abc import Iterator

from bs4 import BeautifulSoup
from bs4.element import Tag

from libs.core.exceptions import ParsingError
from libs.ingestion.normalizers.text_cleaner import collapse_whitespace, extract_text, is_boilerplate
from libs.ingestion.parsers.base import Block, BlockType, FilingParser, ParsedDocument
from libs.ingestion.table_extraction.html_tables import extract_table

_LEAF_BLOCK_TAGS = {"p", "div", "li"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_CONTAINER_CHILD_TAGS = list(_LEAF_BLOCK_TAGS | _HEADING_TAGS | {"table"})


class SecHtmlParser(FilingParser):
    """Parses SEC EDGAR 10-K HTML into an ordered ParsedDocument."""

    def parse(self, raw_content: str | bytes) -> ParsedDocument:
        soup = BeautifulSoup(raw_content, "lxml")
        root = soup.body or soup
        for tag in root.find_all(["script", "style"]):
            tag.decompose()

        blocks: list[Block] = []
        for tag in _walk_blocks(root):
            block = _tag_to_block(tag, index=len(blocks))
            if block is not None:
                blocks.append(block)

        if not blocks:
            raise ParsingError("No content blocks could be extracted from the filing HTML")

        return ParsedDocument(blocks=blocks)


def _walk_blocks(node: Tag) -> Iterator[Tag]:
    """Yields leaf block-level tags and table tags, in document order.

    A "leaf" block has no nested block-level descendants, which avoids
    emitting the same text twice for a <div> that merely wraps a <p>.
    """
    for child in node.find_all(recursive=False):
        if not isinstance(child, Tag):
            continue
        if child.name == "table":
            yield child
        elif child.name in _HEADING_TAGS:
            yield child
        elif child.name in _LEAF_BLOCK_TAGS:
            if child.find(_CONTAINER_CHILD_TAGS) is not None:
                yield from _walk_blocks(child)
            else:
                yield child
        else:
            yield from _walk_blocks(child)


def _tag_to_block(tag: Tag, index: int) -> Block | None:
    if tag.name == "table":
        table = extract_table(tag)
        if not table.rows and not table.header:
            return None
        if not table.rows:
            # A single non-empty row with no body rows is real EDGAR filings'
            # way of laying out one line of text (a heading, a checkbox
            # option) in aligned columns rather than a real data table --
            # surface it as prose so section detection can see it.
            text = collapse_whitespace(" ".join(cell for cell in table.header if cell))
            if not text or is_boilerplate(text):
                return None
            return Block(
                index=index, block_type=BlockType.TEXT, text=text,
                is_emphasized=_looks_emphasized(tag),
            )
        return Block(index=index, block_type=BlockType.TABLE, table=table)

    text = extract_text(tag)
    if not text or is_boilerplate(text):
        return None

    is_emphasized = tag.name in _HEADING_TAGS or _looks_emphasized(tag)
    return Block(index=index, block_type=BlockType.TEXT, text=text, is_emphasized=is_emphasized)


def _looks_emphasized(tag: Tag) -> bool:
    for element in (tag, *tag.find_all(True)):
        style = (element.get("style") or "").lower()
        if "bold" in style or "underline" in style:
            return True
    return tag.find(["b", "strong", "u"]) is not None
