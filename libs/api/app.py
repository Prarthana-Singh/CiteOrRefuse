"""FastAPI serving layer over the "cite or refuse" pipeline -- the API
layer named in project_idea.txt's tech stack but not built until now;
every earlier phase was invoked directly via scripts (see
scripts/run_eval.py, scripts/answer_amazon_fixture.py). This wraps the
same `answer()` pipeline function those scripts call, over the same 4
fixture filings the eval harness runs against -- there is no endpoint to
ingest a new filing yet (see README limitations).

Run locally: uvicorn libs.api.app:app --reload
Requires real OPENAI_API_KEY and COHERE_API_KEY in the environment.
"""
from fastapi import Depends, FastAPI

from libs.api.dependencies import (
    get_cohere_client,
    get_openai_client,
    get_qdrant_client,
    get_sparse_model,
)
from libs.api.schemas import AnswerRequest
from libs.generation.pipeline import AnswerResult, answer

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
    return answer(
        request.query,
        qdrant_client,
        openai_client,
        sparse_model,
        cohere_client,
        request.top_k,
    )
