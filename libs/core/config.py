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


settings = ChunkingSettings()
embedding_settings = EmbeddingSettings()
vectorstore_settings = VectorStoreSettings()
