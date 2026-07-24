"""Domain models shared across ingestion, chunking, and downstream modules."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"


class Filing(BaseModel):
    """A single SEC filing being ingested."""

    filing_id: str
    company: str
    cik: str | None = None
    fiscal_year: int | None = None
    form_type: str = "10-K"
    source_path: str


class Table(BaseModel):
    """A structured representation of an HTML <table>, preserved row-by-row."""

    header: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str | None = None

    def to_markdown(self) -> str:
        lines: list[str] = []
        if self.caption:
            lines.append(f"**{self.caption}**")
        if self.header:
            lines.append("| " + " | ".join(self.header) + " |")
            lines.append("| " + " | ".join("---" for _ in self.header) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


class Section(BaseModel):
    """A detected SEC 10-K section (e.g. Item 1A. Risk Factors)."""

    item_code: str
    title: str
    part: str | None = None
    start_block_idx: int
    end_block_idx: int


class Chunk(BaseModel):
    """A single retrievable, citable unit produced by the chunking pipeline."""

    chunk_id: str
    filing_id: str
    chunk_type: ChunkType
    section_item_code: str
    section_title: str
    order_index: int
    char_start: int
    char_end: int
    token_count: int
    text: str
