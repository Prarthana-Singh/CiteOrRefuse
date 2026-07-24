from types import SimpleNamespace

import pytest

from libs.core.config import embedding_settings
from libs.core.models import Chunk, ChunkType, Filing
from libs.embedding.openai_embedder import build_embedding_text, embed_chunks

_FILING = Filing(filing_id="acme-10k", company="Acme Robotics, Inc.", source_path="x.html")

_PROSE_CHUNK = Chunk(
    chunk_id="acme-10k:1:0",
    filing_id="acme-10k",
    chunk_type=ChunkType.TEXT,
    section_item_code="1",
    section_title="Business",
    part="I",
    order_index=0,
    char_start=0,
    char_end=40,
    token_count=8,
    text="We design and manufacture industrial robots.",
)

_TABLE_CHUNK = Chunk(
    chunk_id="acme-10k:8:table:0",
    filing_id="acme-10k",
    chunk_type=ChunkType.TABLE,
    section_item_code="8",
    section_title="Financial Statements and Supplementary Data",
    part="II",
    order_index=1,
    char_start=100,
    char_end=200,
    token_count=30,
    text="| Region | Revenue |\n| --- | --- |\n| North America | $1,200 |\n| EMEA | $800 |",
)


def test_build_embedding_text_prepends_company_and_section_header():
    header_text = build_embedding_text(_PROSE_CHUNK, _FILING)

    assert header_text == (
        "Acme Robotics, Inc. 10-K, 1 Business: "
        "We design and manufacture industrial robots."
    )
    # The header is embedding-input only -- the original chunk is untouched.
    assert _PROSE_CHUNK.text == "We design and manufacture industrial robots."


def test_build_embedding_text_anchors_a_bare_table_chunk_topically():
    """A table chunk of just region names and dollar figures carries no
    topical anchoring on its own -- the header must supply it, without
    altering the original markdown that gets shown for citation.
    """
    header_text = build_embedding_text(_TABLE_CHUNK, _FILING)

    assert header_text.startswith("Acme Robotics, Inc. 10-K, 8 Financial Statements")
    assert header_text.endswith(_TABLE_CHUNK.text)
    assert _TABLE_CHUNK.text.startswith("| Region | Revenue |")


class _FakeEmbeddings:
    def __init__(self):
        self.calls: list[list[str]] = []

    def create(self, model, input, dimensions):
        self.calls.append(list(input))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(i)] * dimensions) for i in range(len(input))]
        )


class _FakeOpenAIClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


def test_embed_chunks_returns_one_vector_per_chunk_in_order():
    client = _FakeOpenAIClient()
    chunks = [_PROSE_CHUNK, _TABLE_CHUNK]

    vectors = embed_chunks(chunks, _FILING, client)

    assert len(vectors) == 2
    assert len(vectors[0]) == embedding_settings.dimensions
    # The fake client returns a distinct vector per input position, so
    # order-preservation is directly observable.
    assert vectors[0] != vectors[1]


def test_embed_chunks_batches_requests(monkeypatch):
    monkeypatch.setattr(embedding_settings, "batch_size", 2)
    client = _FakeOpenAIClient()
    chunks = [_PROSE_CHUNK, _TABLE_CHUNK, _PROSE_CHUNK, _PROSE_CHUNK, _PROSE_CHUNK]

    vectors = embed_chunks(chunks, _FILING, client)

    assert len(vectors) == 5
    # 5 chunks at batch_size=2 -> 3 API calls (2, 2, 1), not one per chunk.
    assert [len(call) for call in client.embeddings.calls] == [2, 2, 1]
