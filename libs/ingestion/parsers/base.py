"""Parser interface all filing formats (HTML now, PDF later) normalize toward."""
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from libs.core.models import Table


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"


class Block(BaseModel):
    """A single ordered content unit extracted from a filing document."""

    index: int
    block_type: BlockType
    text: str = ""
    is_emphasized: bool = False
    table: Table | None = None


class ParsedDocument(BaseModel):
    blocks: list[Block]


class FilingParser(ABC):
    """Turns raw filing content into an ordered ParsedDocument.

    Every implementation normalizes toward the same Block structure so
    section detection, table extraction, and chunking never need to know
    which source format (HTML, eventually PDF) a filing came from.
    """

    @abstractmethod
    def parse(self, raw_content: str) -> ParsedDocument:
        raise NotImplementedError
