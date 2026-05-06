from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

FinishReason = Literal["stop", "length", "cancelled"]


@dataclass
class GenerationChunk:
    text: str
    finish_reason: FinishReason | None = None


class TextGenerator(Protocol):
    def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[GenerationChunk]: ...
