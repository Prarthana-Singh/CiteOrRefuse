"""Request/response schemas for the API layer. `/answer`'s response is
`AnswerResult` itself (`libs.generation.pipeline`) -- already a Pydantic
model with exactly the shape a caller needs, so it's reused directly
rather than wrapped in a second, API-specific model. `/ingest` gets its
own small response model since none of the underlying pipeline's models
(`IndexResult` in `libs.indexing.pipeline`) carry quite the right shape
(it has `points_written`, not `company`/`sections_detected`).
"""
from pydantic import BaseModel


class AnswerRequest(BaseModel):
    query: str
    top_k: int | None = None
    filing_id: str | None = None
    """Scopes retrieval to one filing -- typically a filing_id returned by
    a prior `/ingest` call. Omitted (the default) preserves this
    endpoint's original behavior: answered against the baked-in fixture
    filings only, never against any uploaded filing that isn't explicitly
    named here (see `libs/api/app.py` for how the default set is applied)
    -- an uploaded filing is only ever queryable by callers who know its
    filing_id, not by anyone sending an unscoped query.
    """


class IngestResponse(BaseModel):
    filing_id: str
    company: str
    sections_detected: int
    chunks_indexed: int
