"""FastAPI serving layer over the "cite or refuse" pipeline -- the API
layer named in project_idea.txt's tech stack but not built until now;
every earlier phase was invoked directly via scripts (see
scripts/run_eval.py, scripts/answer_amazon_fixture.py).

`POST /answer` wraps the same `answer()` pipeline those scripts call, over
the same 4 fixture filings the eval harness runs against by default.
`POST /ingest` (Phase 7) adds a real user-facing ingest path on top of
Phase 1/2's existing, unchanged `ingest_filing()` / `chunk_filing()` /
`index_filing()` functions -- HTML uploads only, session-scoped (an
uploaded filing is retrievable only by callers who pass its filing_id back
to `/answer`, never by an unscoped query), no permanent storage, no new
pipeline logic.

Run locally: uvicorn libs.api.app:app --reload
Requires real OPENAI_API_KEY and COHERE_API_KEY in the environment.
"""
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile

from libs.api.dependencies import (
    get_cohere_client,
    get_ingest_rate_limiter,
    get_openai_client,
    get_qdrant_client,
    get_sparse_model,
)
from libs.api.rate_limit import FixedWindowRateLimiter
from libs.api.schemas import AnswerRequest, IngestResponse
from libs.chunking.pipeline import chunk_filing
from libs.core.config import ingest_settings
from libs.core.exceptions import CiteOrRefuseError
from libs.core.models import Filing
from libs.generation.pipeline import AnswerResult, answer
from libs.indexing.fixtures import FILINGS
from libs.indexing.pipeline import index_filing
from libs.ingestion.pipeline import ingest_filing

# The default retrieval scope for an unscoped `/answer` query -- fixed at
# the 4 baked-in demo filings, so an uploaded filing never surfaces in
# anyone's query unless they explicitly pass its filing_id. This is an
# API-layer-only concept (the core `answer()`/`retrieve()` functions stay
# blind to "fixtures vs. uploads" -- see libs/generation/pipeline.py).
_FIXTURE_FILING_IDS = [filing.filing_id for filing in FILINGS]

app = FastAPI(
    title="CiteOrRefuse",
    description="SEC-filing Q&A that either cites a specific retrieved passage or refuses.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResult)
def post_answer(
    request: AnswerRequest,
    openai_client=Depends(get_openai_client),
    qdrant_client=Depends(get_qdrant_client),
    sparse_model=Depends(get_sparse_model),
    cohere_client=Depends(get_cohere_client),
) -> AnswerResult:
    filing_ids = [request.filing_id] if request.filing_id else _FIXTURE_FILING_IDS
    return answer(
        request.query,
        qdrant_client,
        openai_client,
        sparse_model,
        cohere_client,
        request.top_k,
        filing_ids,
    )


@app.post("/ingest", response_model=IngestResponse)
async def post_ingest(
    request: Request,
    file: UploadFile = File(...),
    company: str = Form(...),
    openai_client=Depends(get_openai_client),
    qdrant_client=Depends(get_qdrant_client),
    sparse_model=Depends(get_sparse_model),
    rate_limiter: FixedWindowRateLimiter = Depends(get_ingest_rate_limiter),
) -> IngestResponse:
    """Ingests one uploaded 10-K (.htm/.html only) for this session: parses,
    detects sections, chunks, and indexes it into the same shared
    collection `/answer` searches -- but it's only ever retrievable by a
    caller who passes the returned `filing_id` back to `/answer`.

    The existing `UnsupportedDocumentTypeError` guard (see
    `libs.ingestion.pipeline`) does real work here, not just an internal
    safety net: this endpoint always ingests as `form_type="10-K"`, so any
    non-10-K upload (a 10-Q, an 8-K) is rejected by that guard, not by new
    logic added here.
    """
    client_key = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_key):
        raise HTTPException(
            status_code=429,
            detail="Too many ingestion requests -- please wait and try again.",
        )

    filename = file.filename or ""
    if not filename.lower().endswith((".htm", ".html")):
        raise HTTPException(
            status_code=400,
            detail="Only .htm/.html filing uploads are supported.",
        )

    content = await file.read(ingest_settings.max_upload_bytes + 1)
    if len(content) > ingest_settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {ingest_settings.max_upload_bytes}-byte upload limit.",
        )

    filing_id = f"upload-{uuid4().hex[:12]}"
    with tempfile.NamedTemporaryFile(suffix=".htm", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        filing = Filing(filing_id=filing_id, company=company, source_path=tmp_path)
        try:
            blocks, sections = ingest_filing(filing)
            chunks = chunk_filing(filing, blocks, sections)
        except CiteOrRefuseError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        index_filing(chunks, filing, openai_client, qdrant_client, sparse_model)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        filing_id=filing_id,
        company=company,
        sections_detected=len(sections),
        chunks_indexed=len(chunks),
    )
