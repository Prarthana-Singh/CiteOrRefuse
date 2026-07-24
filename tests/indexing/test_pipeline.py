import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from libs.chunking.pipeline import chunk_filing_from_source
from libs.core.config import embedding_settings, vectorstore_settings
from libs.core.models import Filing
from libs.indexing.pipeline import index_filing

_FIXTURES_DIR = Path(__file__).parents[2] / "data" / "fixtures"
_DIM = 32


class _FakeEmbeddings:
    def create(self, model, input, dimensions):
        # A deterministic vector per input string, not just its position --
        # so identical chunk text always embeds to the identical vector
        # across separate calls, matching real embedding-model behavior
        # closely enough for these tests.
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=_deterministic_vector(text)) for text in input
            ]
        )


def _deterministic_vector(text: str) -> list[float]:
    # A seeded-random vector, not one-hot: with 232 real chunks, a small
    # discrete one-hot space (e.g. 8 buckets) guarantees collisions between
    # unrelated chunks, which cosine distance can't then tell apart.
    # `random.Random` seeded directly from the string keeps this
    # deterministic (identical text -> identical vector) while giving
    # near-orthogonal directions between different texts even at this
    # modest dimension.
    rng = random.Random(text)
    return [rng.uniform(-1, 1) for _ in range(_DIM)]


class _FakeOpenAIClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


@pytest.fixture(scope="module")
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)


@pytest.fixture
def qdrant_client(monkeypatch) -> QdrantClient:
    monkeypatch.setattr(embedding_settings, "dimensions", _DIM)
    return QdrantClient(":memory:")


@pytest.fixture
def amazon_filing_and_chunks():
    filing = Filing(
        filing_id="amzn-2025-10k",
        company="Amazon.com, Inc.",
        cik="0001018724",
        fiscal_year=2025,
        source_path=str(_FIXTURES_DIR / "amzn-20251231.html"),
    )
    chunks = chunk_filing_from_source(filing)
    return filing, chunks


def test_index_filing_writes_one_point_per_chunk(
    amazon_filing_and_chunks, qdrant_client, sparse_model
):
    filing, chunks = amazon_filing_and_chunks

    result = index_filing(chunks, filing, _FakeOpenAIClient(), qdrant_client, sparse_model)

    assert result.filing_id == "amzn-2025-10k"
    assert result.points_written == len(chunks)
    assert qdrant_client.count(vectorstore_settings.collection_name).count == len(chunks)


def test_index_filing_is_idempotent(amazon_filing_and_chunks, qdrant_client, sparse_model):
    filing, chunks = amazon_filing_and_chunks
    openai_client = _FakeOpenAIClient()

    first = index_filing(chunks, filing, openai_client, qdrant_client, sparse_model)
    second = index_filing(chunks, filing, openai_client, qdrant_client, sparse_model)

    assert first.points_written == second.points_written == len(chunks)
    # Re-indexing must overwrite existing points, not add alongside them.
    assert qdrant_client.count(vectorstore_settings.collection_name).count == len(chunks)


def test_retrieved_point_payload_reconstructs_a_valid_citation(
    amazon_filing_and_chunks, qdrant_client, sparse_model
):
    """A retrieved point's payload alone -- no second lookup -- must be
    enough to show the user where a claim came from: which filing, which
    company, which Item/Part, and the exact text plus its position in the
    section it was chunked from.
    """
    from libs.vectorstore.qdrant_store import search_dense

    filing, chunks = amazon_filing_and_chunks
    index_filing(chunks, filing, _FakeOpenAIClient(), qdrant_client, sparse_model)

    item7_chunk = next(c for c in chunks if c.section_item_code == "7")
    query_vector = _deterministic_vector(
        f"{filing.company} {filing.form_type}, "
        f"{item7_chunk.section_item_code} {item7_chunk.section_title}: {item7_chunk.text}"
    )

    results = search_dense(qdrant_client, query_vector, top_k=1)
    payload = results[0].payload

    assert payload["company"] == "Amazon.com, Inc."
    assert payload["section_item_code"] == "7"
    assert payload["part"] == "II"
    assert payload["char_end"] > payload["char_start"]
    assert len(payload["text"]) > 0
