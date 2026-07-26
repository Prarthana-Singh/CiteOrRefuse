"""The 4 real filings (3 10-Ks + 1 10-Q) this project's demo dataset is
built from, and the shared indexing helper for them -- used by both
`scripts/run_eval.py` and the API's startup, so the two never disagree
about which filings back the demo.
"""
from pathlib import Path

from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient

from libs.chunking.pipeline import chunk_filing_from_source
from libs.core.models import Filing
from libs.indexing.pipeline import index_filing

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "data" / "fixtures"

FILINGS = [
    Filing(
        filing_id="amzn-2025-10k",
        company="Amazon.com, Inc.",
        cik="0001018724",
        fiscal_year=2025,
        form_type="10-K",
        source_path=str(_FIXTURES_DIR / "amzn-20251231.html"),
    ),
    Filing(
        filing_id="tsla-2025-10k",
        company="Tesla, Inc.",
        cik="0001318605",
        fiscal_year=2025,
        form_type="10-K",
        source_path=str(_FIXTURES_DIR / "tsla-20251231.html"),
    ),
    Filing(
        filing_id="msft-2025-10k",
        company="Microsoft Corporation",
        cik="0000789019",
        fiscal_year=2025,
        form_type="10-K",
        source_path=str(_FIXTURES_DIR / "10-K.html"),
    ),
    Filing(
        filing_id="aapl-2026-10q",
        company="Apple Inc.",
        cik="0000320193",
        fiscal_year=2026,
        form_type="10-Q",
        source_path=str(_FIXTURES_DIR / "aapl-20260328.html"),
    ),
]


def index_all_fixtures(
    openai_client: OpenAI, qdrant_client: QdrantClient, sparse_model: SparseTextEmbedding
) -> None:
    for filing in FILINGS:
        chunks = chunk_filing_from_source(filing)
        index_filing(chunks, filing, openai_client, qdrant_client, sparse_model)
