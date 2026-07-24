"""Exceptions raised across the ingestion and chunking pipeline."""


class CiteOrRefuseError(Exception):
    """Base class for all pipeline errors."""


class ParsingError(CiteOrRefuseError):
    """Raised when a filing document cannot be parsed into blocks."""


class SectionDetectionError(CiteOrRefuseError):
    """Raised when the SEC 10-K section structure cannot be determined."""


class ChunkingError(CiteOrRefuseError):
    """Raised when chunking fails to produce valid, bounded chunks."""
