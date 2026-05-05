from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CBSERVER_", env_file=".env", extra="ignore")

    model: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    dtype: str = Field(default="bfloat16")
    device: str = Field(default="cuda")
    attn_implementation: str = Field(default="eager")

    max_seq_len: int = Field(default=2048, ge=64)
    max_num_batched_tokens: int = Field(default=2048, ge=64)
    max_num_seqs: int = Field(default=64, ge=1)
    block_size: int = Field(default=16, ge=1)
    num_blocks: int = Field(default=1024, ge=16)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")


def get_settings() -> Settings:
    return Settings()
