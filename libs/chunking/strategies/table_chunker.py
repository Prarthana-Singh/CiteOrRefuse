"""Chunks a Table, guaranteeing a chunk boundary never falls mid-row."""
from libs.core.config import settings
from libs.core.models import Chunk, ChunkType, Table
from libs.chunking.token_utils import count_tokens


def chunk_table(
    table: Table,
    filing_id: str,
    section_item_code: str,
    section_title: str,
    start_order_index: int,
    char_start: int,
    char_end: int,
) -> list[Chunk]:
    """Chunks a table into one or more markdown chunks.

    If the whole table fits within `max_tokens` it becomes a single chunk.
    Otherwise it is split into row-groups, with the header row repeated in
    every sub-chunk so each one is readable on its own -- a row is never
    split across two chunks.
    """
    max_tokens = settings.max_tokens
    full_markdown = table.to_markdown()

    if not table.rows or count_tokens(full_markdown) <= max_tokens:
        return [
            _make_chunk(full_markdown, filing_id, section_item_code, section_title,
                        start_order_index, char_start, char_end)
        ]

    chunks: list[Chunk] = []
    order_index = start_order_index
    current_rows: list[list[str]] = []

    def markdown_for(rows: list[list[str]]) -> str:
        return Table(header=table.header, rows=rows, caption=table.caption).to_markdown()

    def flush() -> None:
        nonlocal order_index, current_rows
        if not current_rows:
            return
        chunks.append(
            _make_chunk(markdown_for(current_rows), filing_id, section_item_code, section_title,
                        order_index, char_start, char_end)
        )
        order_index += 1
        current_rows = []

    for row in table.rows:
        candidate_rows = current_rows + [row]
        # Check the *actual* rendered markdown, not an estimate: pipe/newline
        # formatting overhead means summed per-cell token counts undercount
        # the real chunk size.
        if current_rows and count_tokens(markdown_for(candidate_rows)) > max_tokens:
            flush()
            candidate_rows = [row]
        current_rows = candidate_rows

    flush()
    return chunks


def _make_chunk(
    markdown: str,
    filing_id: str,
    section_item_code: str,
    section_title: str,
    order_index: int,
    char_start: int,
    char_end: int,
) -> Chunk:
    return Chunk(
        chunk_id=f"{filing_id}:{section_item_code}:table:{order_index}",
        filing_id=filing_id,
        chunk_type=ChunkType.TABLE,
        section_item_code=section_item_code,
        section_title=section_title,
        order_index=order_index,
        char_start=char_start,
        char_end=char_end,
        token_count=count_tokens(markdown),
        text=markdown,
    )
