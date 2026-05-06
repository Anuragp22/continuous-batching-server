import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from cbserver.engine.protocols import FinishReason


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("prompt must not be empty")
        return value


class Choice(BaseModel):
    text: str
    index: int = 0
    finish_reason: FinishReason | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _new_completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


def _now() -> int:
    return int(time.time())


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=_new_completion_id)
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[Choice]
    usage: Usage


class StreamChunk(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[Choice]
