from pathlib import Path

import pytest

from libs.core.config import settings
from libs.core.models import ChunkType, Filing
from libs.ingestion.pipeline import ingest_filing
from libs.chunking.pipeline import assemble_section_text, chunk_filing, chunk_filing_from_source

FIXTURE_PATH = Path(__file__).parents[2] / "data" / "fixtures" / "sample_10k_excerpt.htm"


@pytest.fixture(scope="module")
def filing() -> Filing:
    return Filing(
        filing_id="acme-2023-10k",
        company="Acme Robotics, Inc.",
        cik="0001234567",
        fiscal_year=2023,
        form_type="10-K",
        source_path=str(FIXTURE_PATH),
    )


def test_end_to_end_chunking_produces_expected_sections(filing):
    blocks, sections = ingest_filing(filing)
    chunks = chunk_filing(filing, blocks, sections)

    section_codes = {c.section_item_code for c in chunks}
    assert section_codes == {"1", "1A", "7", "8"}
    assert len(chunks) > 0


def test_table_chunk_is_present_and_clean(filing):
    chunks = chunk_filing_from_source(filing)
    table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]

    assert len(table_chunks) == 1
    table_text = table_chunks[0].text
    assert "Total revenue" in table_text
    assert "$412,300" in table_text
    assert table_chunks[0].section_item_code == "7"
    # The empty spacer row from the source HTML must not appear as a blank data row.
    assert "|  |  |  |  |" not in table_text


def test_all_chunks_respect_token_budget(filing):
    chunks = chunk_filing_from_source(filing)
    assert all(c.token_count <= settings.max_tokens for c in chunks)
    assert all(c.char_start <= c.char_end for c in chunks)


def test_text_chunk_offsets_match_reconstructed_section_text(filing):
    blocks, sections = ingest_filing(filing)
    chunks = chunk_filing(filing, blocks, sections)

    for section in sections:
        section_blocks = [
            b for b in blocks if section.start_block_idx <= b.index <= section.end_block_idx
        ]
        canonical_text, _ = assemble_section_text(section_blocks)

        text_chunks = [
            c
            for c in chunks
            if c.section_item_code == section.item_code and c.chunk_type == ChunkType.TEXT
        ]
        for chunk in text_chunks:
            assert canonical_text[chunk.char_start : chunk.char_end] == chunk.text


def test_chunk_ids_are_unique(filing):
    chunks = chunk_filing_from_source(filing)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
