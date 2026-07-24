"""Runs the ingestion + chunking pipeline end-to-end against the sample fixture.

Usage: python scripts/demo_ingest_fixture.py
"""
import json
from pathlib import Path

from libs.core.models import Filing
from libs.chunking.pipeline import chunk_filing_from_source

FIXTURE_PATH = Path(__file__).parent.parent / "data" / "fixtures" / "sample_10k_excerpt.htm"


def main() -> None:
    filing = Filing(
        filing_id="acme-2023-10k",
        company="Acme Robotics, Inc.",
        cik="0001234567",
        fiscal_year=2023,
        form_type="10-K",
        source_path=str(FIXTURE_PATH),
    )

    chunks = chunk_filing_from_source(filing)

    print(f"\nProduced {len(chunks)} chunks from {FIXTURE_PATH.name}\n")
    for chunk in chunks:
        preview = chunk.text.replace("\n", " ")[:80]
        print(
            f"[{chunk.order_index:02d}] {chunk.chunk_type.value:5s} "
            f"Item {chunk.section_item_code:>3s} | tokens={chunk.token_count:4d} | "
            f"chars=({chunk.char_start},{chunk.char_end}) | {preview}..."
        )

    out_path = Path(r"C:\Users\HP\AppData\Local\Temp\claude\c--Users-HP-CiteOrRefuse\d15fbe79-292c-4fcf-af2c-caafcc7c54c6\scratchpad\sample_10k_excerpt.chunks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([c.model_dump(mode="json") for c in chunks], indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote chunk JSON to {out_path}")


if __name__ == "__main__":
    main()
