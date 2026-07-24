from bs4 import BeautifulSoup

from libs.ingestion.table_extraction.html_tables import extract_table


def _table_tag(html: str):
    return BeautifulSoup(html, "lxml").find("table")


def test_extracts_header_and_rows():
    html = """
    <table>
      <tr><th>Metric</th><th>FY23</th><th>FY22</th></tr>
      <tr><td>Revenue</td><td>$100</td><td>$80</td></tr>
      <tr><td>Net income</td><td>$10</td><td>$5</td></tr>
    </table>
    """
    table = extract_table(_table_tag(html))
    assert table.header == ["Metric", "FY23", "FY22"]
    assert table.rows == [["Revenue", "$100", "$80"], ["Net income", "$10", "$5"]]


def test_drops_fully_empty_spacer_rows():
    html = """
    <table>
      <tr><td>Metric</td><td>FY23</td></tr>
      <tr><td></td><td></td></tr>
      <tr><td>Revenue</td><td>$100</td></tr>
    </table>
    """
    table = extract_table(_table_tag(html))
    assert table.rows == [["Revenue", "$100"]]


def test_captures_caption():
    html = """
    <table>
      <caption>Selected Financial Data</caption>
      <tr><td>Metric</td><td>FY23</td></tr>
      <tr><td>Revenue</td><td>$100</td></tr>
    </table>
    """
    table = extract_table(_table_tag(html))
    assert table.caption == "Selected Financial Data"


def test_cell_word_split_across_adjacent_inline_tags_is_not_space_joined():
    """Same class of bug as html_parser.py's leaf-text extraction: a table
    cell whose content is split across adjacent inline tags with no
    whitespace between them in the source must not gain a synthetic space.
    """
    html = """
    <table>
      <tr><td>Metric</td><td><span>Reven</span><span>ue</span></td></tr>
      <tr><td>FY23</td><td>$100</td></tr>
    </table>
    """
    table = extract_table(_table_tag(html))
    assert table.header == ["Metric", "Revenue"]


def test_empty_table_returns_empty_model():
    html = "<table></table>"
    table = extract_table(_table_tag(html))
    assert table.header == []
    assert table.rows == []
