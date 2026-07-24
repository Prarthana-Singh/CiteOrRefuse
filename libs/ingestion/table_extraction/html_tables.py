"""Extracts a structured Table from an HTML <table> element."""
from bs4.element import Tag

from libs.core.models import Table
from libs.ingestion.normalizers.text_cleaner import extract_text


def extract_table(table_tag: Tag) -> Table:
    """Converts a <table> tag into a row-preserving Table model.

    SEC HTML tables are routinely padded with empty <td> spacer cells used
    purely for column alignment; rows that are empty across every cell are
    dropped since they carry no information and would otherwise waste chunk
    budget. The first remaining row is treated as the header.
    """
    caption = None
    if table_tag.caption is not None:
        caption_text = extract_text(table_tag.caption)
        caption = caption_text or None

    all_rows: list[list[str]] = []
    for row in table_tag.find_all("tr"):
        cells = [
            extract_text(cell)
            for cell in row.find_all(["td", "th"])
        ]
        if any(cell for cell in cells):
            all_rows.append(cells)

    if not all_rows:
        return Table(header=[], rows=[], caption=caption)

    header, *body_rows = all_rows
    return Table(header=header, rows=body_rows, caption=caption)
