"""Pipeline-wide configuration defaults, overridable via CITEORREFUSE_ env vars."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CITEORREFUSE_CHUNK_")

    target_tokens: int = 400
    max_tokens: int = 600
    overlap_tokens: int = 60
    tokenizer_encoding: str = "cl100k_base"


settings = ChunkingSettings()
