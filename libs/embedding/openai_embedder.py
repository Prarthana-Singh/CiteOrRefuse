"""Turns Chunks into dense embedding vectors via the OpenAI API."""
from openai import OpenAI

from libs.core.config import embedding_settings
from libs.core.models import Chunk, Filing


def build_embedding_text(chunk: Chunk, filing: Filing) -> str:
    """Builds the text actually sent to the embedding model.

    A bare table chunk (e.g. just region names and dollar figures) carries
    no topical anchoring on its own -- prepending company/section context
    gives the embedding something to place it near semantically. This
    header is embedding-input only: `chunk.text`, what's shown to the user
    for citation, is never modified.
    """
    return (
        f"{filing.company} {filing.form_type}, "
        f"{chunk.section_item_code} {chunk.section_title}: {chunk.text}"
    )


def embed_chunks(chunks: list[Chunk], filing: Filing, client: OpenAI) -> list[list[float]]:
    """Returns one dense vector per chunk, in the same order as `chunks`.

    Batches requests (`embedding_settings.batch_size` per call) rather than
    one call per chunk, since a filing can produce hundreds of chunks.
    `client` is injected rather than constructed here so tests can pass a
    fake/mock without needing a real API key.
    """
    vectors: list[list[float]] = []
    batch_size = embedding_settings.batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        inputs = [build_embedding_text(chunk, filing) for chunk in batch]
        response = client.embeddings.create(
            model=embedding_settings.model,
            input=inputs,
            dimensions=embedding_settings.dimensions,
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors
