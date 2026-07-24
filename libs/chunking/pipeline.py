"""Orchestrates chunking of every detected section of a filing, in order."""
from libs.core.models import Chunk, Filing, Section
from libs.ingestion.parsers.base import Block, BlockType
from libs.ingestion.pipeline import ingest_filing
from libs.chunking.strategies.table_chunker import chunk_table
from libs.chunking.strategies.text_chunker import chunk_text_blocks


def assemble_section_text(section_blocks: list[Block]) -> tuple[str, list[tuple[int, int]]]:
    """Builds the canonical text a section's chunk offsets are anchored to.

    Table blocks are represented by a placeholder span rather than their
    content (their content lives in their own chunk's text), so every block
    still gets a real, non-overlapping (char_start, char_end) into a single
    shared string -- this is what keeps text-chunk and table-chunk offsets
    mutually consistent across a section that mixes prose and tables.
    """
    pieces = [
        block.text if block.block_type == BlockType.TEXT else f"[TABLE #{block.index}]"
        for block in section_blocks
    ]
    text = "\n\n".join(pieces)

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        start = cursor
        end = start + len(piece)
        offsets.append((start, end))
        cursor = end + 2  # "\n\n" join separator
    return text, offsets


def chunk_filing(filing: Filing, blocks: list[Block], sections: list[Section]) -> list[Chunk]:
    """Chunks every detected section into citable, ordered Chunks.

    Consecutive TEXT blocks within a section are packed together by the text
    chunker; TABLE blocks are chunked independently so a table can never be
    merged into the surrounding prose token stream (and never split mid-row).
    All offsets are anchored to the section's `assemble_section_text` output,
    so a text chunk's (char_start, char_end) is always an exact slice of that
    canonical string, table placeholder included.
    """
    all_chunks: list[Chunk] = []
    order_index = 0

    for section in sections:
        section_blocks = [
            b for b in blocks if section.start_block_idx <= b.index <= section.end_block_idx
        ]
        _, block_offsets = assemble_section_text(section_blocks)

        pos = 0
        n = len(section_blocks)
        while pos < n:
            block = section_blocks[pos]

            if block.block_type == BlockType.TEXT:
                run_start = pos
                run_blocks: list[Block] = []
                while pos < n and section_blocks[pos].block_type == BlockType.TEXT:
                    run_blocks.append(section_blocks[pos])
                    pos += 1

                base_offset = block_offsets[run_start][0]
                paragraphs = [b.text for b in run_blocks]
                new_chunks = chunk_text_blocks(
                    paragraphs, filing.filing_id, section.item_code, section.title,
                    order_index, section.part,
                )
                for c in new_chunks:
                    c.char_start += base_offset
                    c.char_end += base_offset
                all_chunks.extend(new_chunks)
                order_index += len(new_chunks)
            else:
                char_start, char_end = block_offsets[pos]
                assert block.table is not None
                table_chunks = chunk_table(
                    block.table,
                    filing.filing_id,
                    section.item_code,
                    section.title,
                    order_index,
                    char_start,
                    char_end,
                    section.part,
                )
                all_chunks.extend(table_chunks)
                order_index += len(table_chunks)
                pos += 1

    return all_chunks


def chunk_filing_from_source(filing: Filing) -> list[Chunk]:
    """Convenience entrypoint: parse + detect sections + chunk in one call."""
    blocks, sections = ingest_filing(filing)
    return chunk_filing(filing, blocks, sections)
