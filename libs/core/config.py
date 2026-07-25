"""Pipeline-wide configuration defaults, overridable via CITEORREFUSE_ env vars."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CITEORREFUSE_CHUNK_")

    target_tokens: int = 400
    max_tokens: int = 600
    overlap_tokens: int = 60
    tokenizer_encoding: str = "cl100k_base"


class EmbeddingSettings(BaseSettings):
    """The OpenAI API key itself is read by the OpenAI SDK from its own
    default `OPENAI_API_KEY` env var, not duplicated here -- client
    construction (and thus the key) is the caller's responsibility, which
    keeps `libs/embedding` free of hidden global state and easy to test
    with an injected fake client.
    """

    model_config = SettingsConfigDict(env_prefix="CITEORREFUSE_EMBED_")

    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100


class VectorStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CITEORREFUSE_VECTORSTORE_")

    collection_name: str = "sec_filings"
    dense_vector_name: str = "dense"
    dense_distance: str = "Cosine"
    sparse_vector_name: str = "bm25"
    sparse_model: str = "Qdrant/bm25"


class RetrievalSettings(BaseSettings):
    """The Cohere API key, like the OpenAI key, is read by the Cohere SDK
    from its own env var, not duplicated here -- see `EmbeddingSettings`."""

    model_config = SettingsConfigDict(env_prefix="CITEORREFUSE_RETRIEVAL_")

    prefetch_limit: int = 20
    candidate_limit: int = 20
    default_top_k: int = 5
    rerank_model: str = "rerank-v3.5"


class GenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CITEORREFUSE_GENERATION_")

    model: str = "gpt-4.1"
    temperature: float = 0.0
    context_token_budget: int = 6000


class GroundednessSettings(BaseSettings):
    """The gate that makes this "cite or refuse": every claim in a
    generated answer must resolve to a chunk_id that was actually
    retrieved (checked mechanically, no LLM call needed) AND be judged as
    genuinely supported by that chunk's text (an LLM-as-judge call per
    claim) before the answer is returned. Anything short of that -> refuse.
    """

    model_config = SettingsConfigDict(env_prefix="CITEORREFUSE_GROUNDEDNESS_")

    judge_model: str = "gpt-4.1-mini"
    confidence_threshold: float = 0.7
    refusal_message: str = (
        "I don't have sufficient evidence in the retrieved filings to answer this confidently."
    )


settings = ChunkingSettings()
embedding_settings = EmbeddingSettings()
vectorstore_settings = VectorStoreSettings()
retrieval_settings = RetrievalSettings()
generation_settings = GenerationSettings()
groundedness_settings = GroundednessSettings()
