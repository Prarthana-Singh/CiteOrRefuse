import pytest
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from libs.core.config import embedding_settings, vectorstore_settings
from libs.core.models import Chunk, ChunkType, Filing
from libs.vectorstore.qdrant_store import point_id_for, search_dense, upsert_chunks

_FILING = Filing(filing_id="acme-10k", company="Acme Robotics, Inc.", source_path="x.html")
_DIM = 8


def _chunk(order_index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"acme-10k:1:{order_index}",
        filing_id="acme-10k",
        chunk_type=ChunkType.TEXT,
        section_item_code="1",
        section_title="Business",
        part="I",
        order_index=order_index,
        char_start=order_index * 10,
        char_end=order_index * 10 + len(text),
        token_count=len(text.split()),
        text=text,
    )


def _vector(index: int) -> list[float]:
    # A one-hot-ish vector per index: with cosine distance, vectors that
    # only differ in magnitude (not direction) are indistinguishable, so
    # distinct *directions* are needed to get meaningful nearest-neighbor
    # results in these tests.
    vector = [0.01] * _DIM
    vector[index % _DIM] = 1.0
    return vector


@pytest.fixture(scope="module")
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)


@pytest.fixture
def client(monkeypatch) -> QdrantClient:
    # Keep test vectors tiny (real dimensions=1536 would work too, just
    # slower to no benefit) -- the collection's configured dense size must
    # match whatever's actually upserted.
    monkeypatch.setattr(embedding_settings, "dimensions", _DIM)
    return QdrantClient(":memory:")


def test_point_id_is_deterministic_for_the_same_chunk():
    chunk_a = _chunk(0, "Revenue grew in North America.")
    chunk_b = _chunk(0, "Revenue grew in North America.")

    assert point_id_for(chunk_a) == point_id_for(chunk_b)


def test_point_id_differs_across_filings_and_order_index():
    a = point_id_for(_chunk(0, "text"))
    b = point_id_for(_chunk(1, "text"))
    c = point_id_for(Chunk(**{**_chunk(0, "text").model_dump(), "filing_id": "other", "chunk_id": "other:1:0"}))

    assert len({a, b, c}) == 3


def test_upsert_and_search_round_trip(client, sparse_model):
    chunks = [_chunk(0, "Revenue grew in North America."), _chunk(1, "Costs declined this year.")]
    vectors = [_vector(0), _vector(1)]

    written = upsert_chunks(chunks, vectors, _FILING, client, sparse_model)
    assert written == 2

    results = search_dense(client, _vector(0), top_k=1)
    assert len(results) == 1
    payload = results[0].payload
    assert payload["chunk_id"] == "acme-10k:1:0"
    assert payload["company"] == "Acme Robotics, Inc."
    assert payload["section_item_code"] == "1"
    assert payload["part"] == "I"
    assert payload["text"] == "Revenue grew in North America."
    assert payload["char_start"] == 0


def test_upsert_is_idempotent_and_reflects_the_latest_run(client, sparse_model):
    chunk = _chunk(0, "Original text.")
    upsert_chunks([chunk], [_vector(2)], _FILING, client, sparse_model)
    assert client.count(vectorstore_settings.collection_name).count == 1

    updated_chunk = chunk.model_copy(update={"text": "Updated text."})
    upsert_chunks([updated_chunk], [_vector(2)], _FILING, client, sparse_model)

    # Same chunk_id -> same point ID -> overwrite, not a second point.
    assert client.count(vectorstore_settings.collection_name).count == 1
    results = search_dense(client, _vector(2), top_k=1)
    assert results[0].payload["text"] == "Updated text."


def test_upsert_empty_chunk_list_is_a_noop(client, sparse_model):
    written = upsert_chunks([], [], _FILING, client, sparse_model)
    assert written == 0
