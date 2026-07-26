"""Request schema for the /answer endpoint. The response is `AnswerResult`
itself (`libs.generation.pipeline`) -- already a Pydantic model with
exactly the shape a caller needs, so it's reused directly rather than
wrapped in a second, API-specific model.
"""
from pydantic import BaseModel


class AnswerRequest(BaseModel):
    query: str
    top_k: int | None = None
