"""Generates an answer with claim-level citations via OpenAI structured outputs."""
from openai import OpenAI
from pydantic import BaseModel, Field

from libs.core.config import generation_settings

_SYSTEM_PROMPT = (
    "You are a financial research assistant answering questions about SEC "
    "filings using ONLY the provided context. Every factual claim in your "
    "answer must be backed by a specific chunk from the context, cited by "
    "its exact chunk_id. Never cite a chunk_id that is not present in the "
    "context. If the context does not contain enough information to answer "
    "the question, say so in `answer` and return an empty `claims` list -- "
    "do not guess or use outside knowledge."
)


class Claim(BaseModel):
    """One factual claim within the generated answer, attributed to the
    specific source chunk it's grounded on."""

    text: str = Field(description="The claim, as a standalone factual statement.")
    chunk_id: str = Field(description="The chunk_id (from the context) this claim is based on.")


class GeneratedAnswer(BaseModel):
    """The LLM's structured response: a natural-language answer plus its
    claim-level attribution breakdown, which the groundedness gate verifies
    claim-by-claim before this is ever shown to a user.
    """

    answer: str
    claims: list[Claim]


def generate_answer(query_text: str, context: str, client: OpenAI) -> GeneratedAnswer:
    """Calls the LLM with `context` (see `libs.generation.context_builder.build_context`)
    and returns its structured, claim-attributed answer.

    `client` is injected, never constructed here, so tests never need a
    real key -- same pattern as `libs.embedding.openai_embedder.embed_chunks`.
    """
    completion = client.chat.completions.parse(
        model=generation_settings.model,
        temperature=generation_settings.temperature,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query_text}"},
        ],
        response_format=GeneratedAnswer,
    )
    return completion.choices[0].message.parsed
