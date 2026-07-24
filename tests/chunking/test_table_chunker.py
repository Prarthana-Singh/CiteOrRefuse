from libs.core.config import settings
from libs.core.models import Table
from libs.chunking.strategies.table_chunker import chunk_table

FILING_ID = "acme-2023-10k"


def test_small_table_becomes_a_single_chunk():
    table = Table(
        header=["Metric", "FY23", "FY22"],
        rows=[["Total revenue", "$412,300", "$338,100"], ["Gross profit", "$168,900", "$132,400"]],
        caption="Selected Financial Data",
    )
    chunks = chunk_table(table, FILING_ID, "7", "MD&A", 0, char_start=100, char_end=100)

    assert len(chunks) == 1
    assert "Total revenue" in chunks[0].text
    assert "Gross profit" in chunks[0].text
    assert "Selected Financial Data" in chunks[0].text


def test_large_table_splits_by_row_group_with_repeated_header():
    header = ["Row Label", "Value"]
    rows = [[f"RowLabel{i}", f"{i * 10}"] for i in range(300)]
    table = Table(header=header, rows=rows, caption=None)

    chunks = chunk_table(table, FILING_ID, "7", "MD&A", 0, char_start=0, char_end=0)

    assert len(chunks) > 1
    assert all(c.token_count <= settings.max_tokens for c in chunks)

    # Header must be present (repeated) in every sub-chunk.
    assert all("Row Label" in c.text for c in chunks)

    # Every original data row appears in exactly one chunk -- no row lost, none duplicated.
    # Matched as the full rendered row (not a bare label) since e.g. "RowLabel1" is
    # a substring of "RowLabel10", "RowLabel100", etc.
    for i in range(300):
        row_markdown = f"| RowLabel{i} | {i * 10} |"
        occurrences = sum(c.text.count(row_markdown) for c in chunks)
        assert occurrences == 1


def test_empty_table_still_produces_one_chunk():
    table = Table(header=[], rows=[], caption=None)
    chunks = chunk_table(table, FILING_ID, "8", "Financial Statements", 0, 0, 0)
    assert len(chunks) == 1
