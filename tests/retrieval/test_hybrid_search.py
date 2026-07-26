import random
from types import SimpleNamespace

import pytest
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from libs.core.config import embedding_settings, retrieval_settings, vectorstore_settings
from libs.core.models import Chunk, ChunkType, Filing
from libs.retrieval.hybrid_search import embed_query, hybrid_search
from libs.vectorstore.qdrant_store import upsert_chunks

_FILING = Filing(filing_id="acme-10k", company="Acme Robotics, Inc.", source_path="x.html")
_DIM = 32


def _deterministic_vector(text: str) -> list[float]:
    # Same scheme as tests/indexing/test_pipeline.py: seeded on the text
    # itself, so identical text -> identical vector, near-orthogonal
    # directions between different texts even at this modest dimension.
    rng = random.Random(text)
    return [rng.uniform(-1, 1) for _ in range(_DIM)]


class _FakeEmbeddings:
    def __init__(self):
        self.calls: list[list[str]] = []

    def create(self, model, input, dimensions):
        self.calls.append(list(input))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=_deterministic_vector(text)) for text in input]
        )


class _FakeOpenAIClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


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


@pytest.fixture(scope="module")
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)


@pytest.fixture
def seeded_client(monkeypatch, sparse_model) -> QdrantClient:
    monkeypatch.setattr(embedding_settings, "dimensions", _DIM)
    client = QdrantClient(":memory:")
    chunks = [
        _chunk(0, "North America net sales grew twelve percent this year."),
        _chunk(1, "International segment revenue declined due to currency effects."),
        _chunk(2, "AWS operating income increased on higher utilization."),
    ]
    vectors = [_deterministic_vector(build_text(c)) for c in chunks]
    upsert_chunks(chunks, vectors, _FILING, client, sparse_model)
    return client


def build_text(chunk: Chunk) -> str:
    return f"{_FILING.company} {_FILING.form_type}, {chunk.section_item_code} {chunk.section_title}: {chunk.text}"


def test_embed_query_sends_the_raw_query_text_with_no_context_header(sparse_model):
    """A query is already semantically complete -- unlike a chunk, it must
    not get the "{company} {form_type}, ..." header prepended.
    """
    fake_client = _FakeOpenAIClient()

    embed_query("North America net sales", fake_client, sparse_model)

    assert fake_client.embeddings.calls == [["North America net sales"]]


def test_embed_query_returns_a_dense_vector_and_a_sparse_vector(sparse_model):
    fake_client = _FakeOpenAIClient()

    dense_vector, sparse_vector = embed_query("net sales", fake_client, sparse_model)

    assert len(dense_vector) == _DIM
    assert len(sparse_vector.indices) == len(sparse_vector.values)
    assert len(sparse_vector.indices) > 0


def test_hybrid_search_returns_fused_candidates_with_full_payload(seeded_client, sparse_model):
    fake_client = _FakeOpenAIClient()

    results = hybrid_search("North America net sales", seeded_client, fake_client, sparse_model)

    assert len(results) == 3
    assert all("text" in r.payload for r in results)
    assert all(r.payload["company"] == "Acme Robotics, Inc." for r in results)


def test_hybrid_search_respects_the_limit_argument(seeded_client, sparse_model):
    fake_client = _FakeOpenAIClient()

    results = hybrid_search(
        "North America net sales", seeded_client, fake_client, sparse_model, limit=1
    )

    assert len(results) == 1


_OTHER_FILING = Filing(filing_id="upload-xyz", company="Other Co.", source_path="y.html")


@pytest.fixture
def seeded_client_two_filings(monkeypatch, sparse_model) -> QdrantClient:
    """Two filings in the same collection -- acme-10k (the default fixture
    set here) plus a second, unrelated one -- so filing_ids scoping can be
    verified to actually exclude the other filing's points, not just
    happen to return fewer results by coincidence.
    """
    monkeypatch.setattr(embedding_settings, "dimensions", _DIM)
    client = QdrantClient(":memory:")

    acme_chunks = [_chunk(0, "North America net sales grew twelve percent this year.")]
    acme_vectors = [_deterministic_vector(build_text(c)) for c in acme_chunks]
    upsert_chunks(acme_chunks, acme_vectors, _FILING, client, sparse_model)

    other_chunk = Chunk(
        chunk_id="upload-xyz:1:0",
        filing_id="upload-xyz",
        chunk_type=ChunkType.TEXT,
        section_item_code="1",
        section_title="Business",
        part="I",
        order_index=0,
        char_start=0,
        char_end=10,
        token_count=5,
        text="North America net sales also grew for this other company.",
    )
    other_vector = [_deterministic_vector(
        f"{_OTHER_FILING.company} {_OTHER_FILING.form_type}, 1 Business: {other_chunk.text}"
    )]
    upsert_chunks([other_chunk], other_vector, _OTHER_FILING, client, sparse_model)

    return client


def test_hybrid_search_with_no_filing_ids_searches_the_whole_collection(
    seeded_client_two_filings, sparse_model
):
    fake_client = _FakeOpenAIClient()

    results = hybrid_search(
        "North America net sales", seeded_client_two_filings, fake_client, sparse_model
    )

    filing_ids_seen = {r.payload["filing_id"] for r in results}
    assert filing_ids_seen == {"acme-10k", "upload-xyz"}


def test_hybrid_search_with_filing_ids_excludes_other_filings(
    seeded_client_two_filings, sparse_model
):
    fake_client = _FakeOpenAIClient()

    results = hybrid_search(
        "North America net sales",
        seeded_client_two_filings,
        fake_client,
        sparse_model,
        filing_ids=["acme-10k"],
    )

    assert len(results) > 0
    assert all(r.payload["filing_id"] == "acme-10k" for r in results)
