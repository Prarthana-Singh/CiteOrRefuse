"""Text normalization applied to every block extracted from a filing."""
import re

_WHITESPACE_RE = re.compile(r"[ \t\xa0]+")

_BOILERPLATE_PATTERNS = [
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^\d+$"),  # a standalone page number
    re.compile(r"^page\s+\d+$", re.IGNORECASE),
]


def collapse_whitespace(text: str) -> str:
    """Collapses runs of spaces/tabs/nbsp into a single space and trims ends."""
    text = text.replace("\xa0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def is_boilerplate(text: str) -> bool:
    """Flags common SEC filing chrome (running headers, bare page numbers)."""
    stripped = text.strip()
    return any(pattern.match(stripped) for pattern in _BOILERPLATE_PATTERNS)
