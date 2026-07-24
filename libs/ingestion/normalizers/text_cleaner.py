"""Text normalization applied to every block extracted from a filing."""
import re

from bs4.element import Tag

_WHITESPACE_RE = re.compile(r"\s+")

_BOILERPLATE_PATTERNS = [
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^\d+$"),  # a standalone page number
    re.compile(r"^page\s+\d+$", re.IGNORECASE),
]


def collapse_whitespace(text: str) -> str:
    """Collapses runs of any whitespace (spaces/tabs/nbsp/newlines) into a
    single space and trims ends."""
    text = text.replace("\xa0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def extract_text(tag: Tag) -> str:
    """Extracts a tag's text faithfully to its source whitespace.

    Real EDGAR filings routinely split a single word across two adjacent
    inline tags with no whitespace between them in the source (e.g.
    "<span>STATE</span><span>MENTS</span>" for "STATEMENTS", likely a
    pagination/reflow artifact). bs4's `get_text(separator=" ")` -- needed
    to keep genuinely separate text runs from fusing together, since
    `strip=True` strips whitespace from each individual text fragment
    before joining -- inserts a space at every such tag boundary
    regardless, breaking words apart. `<br>` tags contribute no text of
    their own, so they still need an explicit line-break marker. Mutating
    the tag in place to swap `<br>` for a "\\n" text node and extracting
    with no separator (falling back on `collapse_whitespace` for
    normalization) preserves both cases correctly.
    """
    for br in tag.find_all("br"):
        br.replace_with("\n")
    return collapse_whitespace(tag.get_text(strip=False))


def is_boilerplate(text: str) -> bool:
    """Flags common SEC filing chrome (running headers, bare page numbers)."""
    stripped = text.strip()
    return any(pattern.match(stripped) for pattern in _BOILERPLATE_PATTERNS)
