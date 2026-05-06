import asyncio
from collections.abc import AsyncIterator

from cbserver.engine.protocols import GenerationChunk


class MockGenerator:
    def __init__(self, delay_per_token: float = 0.0) -> None:
        self._delay = delay_per_token

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        for index in range(max_tokens):
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            text = f"tok{index} "
            if stop and any(needle in text for needle in stop):
                yield GenerationChunk(text="", finish_reason="stop")
                return
            yield GenerationChunk(text=text, finish_reason=None)
        yield GenerationChunk(text="", finish_reason="length")
