"""Constructs the pipeline's external clients once per process (the same
composition-root responsibility `scripts/run_eval.py`'s `main()` has) and
wires them through FastAPI's dependency injection instead of a script's
argument list. Each provider is independently overridable via
`app.dependency_overrides` in tests, so route tests never need real
API keys or a real Qdrant instance -- same guarantee every other layer's
test suite already has.

The in-memory Qdrant collection is indexed with the 4 fixture filings
(`libs.indexing.fixtures`) once, the first time `get_qdrant_client` is
resolved -- there is no "ingest a new filing" endpoint yet (see README
limitations); this serves the same known demo dataset the eval harness
runs against.
"""
from functools import lru_cache

import cohere
from fastembed import SparseTextEmbedding
from fastapi import Depends
from openai import OpenAI
from qdrant_client import QdrantClient

from libs.api.rate_limit import FixedWindowRateLimiter
from libs.core.config import ingest_settings, vectorstore_settings
from libs.indexing.fixtures import index_all_fixtures


@lru_cache
def get_openai_client() -> OpenAI:
    return OpenAI()


@lru_cache
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=vectorstore_settings.sparse_model)


@lru_cache
def get_cohere_client() -> cohere.ClientV2:
    return cohere.ClientV2()


@lru_cache
def get_qdrant_client(
    openai_client: OpenAI = Depends(get_openai_client),
    sparse_model: SparseTextEmbedding = Depends(get_sparse_model),
) -> QdrantClient:
    client = QdrantClient(":memory:")
    index_all_fixtures(openai_client, client, sparse_model)
    return client


@lru_cache
def get_ingest_rate_limiter() -> FixedWindowRateLimiter:
    return FixedWindowRateLimiter(
        max_calls=ingest_settings.rate_limit_max_calls,
        window_seconds=ingest_settings.rate_limit_window_seconds,
    )
