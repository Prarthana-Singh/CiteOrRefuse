"""Exceptions raised across the ingestion and chunking pipeline."""


class CiteOrRefuseError(Exception):
    """Base class for all pipeline errors."""


class ParsingError(CiteOrRefuseError):
    """Raised when a filing document cannot be parsed into blocks."""


class UnsupportedDocumentTypeError(CiteOrRefuseError):
    """Raised when a filing's declared SEC form type doesn't match its
    actual `dei:DocumentType`, e.g. a `Filing(form_type="10-K")` pointed at
    a 10-Q. Section detection is tuned to the 10-K Item 1-16 structure and
    will silently produce a plausible-looking but wrong result for other
    form types, so this must be caught before it runs, not after."""


class SectionDetectionError(CiteOrRefuseError):
    """Raised when the SEC 10-K section structure cannot be determined."""


class ChunkingError(CiteOrRefuseError):
    """Raised when chunking fails to produce valid, bounded chunks."""
