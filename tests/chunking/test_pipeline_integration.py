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


_FIXTURES_DIR = Path(__file__).parents[2] / "data" / "fixtures"


@pytest.mark.parametrize(
    "fixture_name,expected_chunk_count,expected_item8_chunk_count",
    [
        ("tsla-20251231.html", 289, 161),
        ("10-K.html", 285, 141),  # Microsoft FY2025 10-K (msft-20250630.htm)
    ],
)
def test_real_filing_chunk_counts(fixture_name, expected_chunk_count, expected_item8_chunk_count):
    """Pins total and per-section chunk counts against real EDGAR filings.

    Item 8 (Financial Statements) is checked specifically because it's the
    section a running-header pagination bug fragmented into ~40 spurious
    sections -- if that regresses, chunk counts across dozens of bogus
    "sections" would shift in a way a total-only assertion could still miss.
    """
    real_filing = Filing(
        filing_id=fixture_name,
        company="Test Co",
        source_path=str(_FIXTURES_DIR / fixture_name),
    )
    chunks = chunk_filing_from_source(real_filing)

    assert len(chunks) == expected_chunk_count
    item8_chunks = [c for c in chunks if c.section_item_code == "8"]
    assert len(item8_chunks) == expected_item8_chunk_count


def test_real_10q_filing_chunk_ids_stay_unique_despite_item_code_reuse():
    """A 10-Q reuses Item codes 1-4 across Part I and Part II with
    different meanings (e.g. Part I Item 1 is "Financial Statements", Part
    II Item 1 is "Legal Proceedings"). `chunk_id` is built from
    `section_item_code`, so this guards against the two same-numbered
    sections ever producing colliding chunk IDs -- protected today by
    `order_index` in `chunk_filing` being a filing-global counter rather
    than one that resets per section.
    """
    real_filing = Filing(
        filing_id="aapl-10q",
        company="Apple Inc.",
        form_type="10-Q",
        source_path=str(_FIXTURES_DIR / "aapl-20260328.html"),
    )
    chunks = chunk_filing_from_source(real_filing)

    assert len(chunks) == 86
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
