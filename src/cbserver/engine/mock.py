import asyncio
from collections.abc import AsyncIterator

from cbserver.engine.protocols import GenerationChunk, StopAwareEmitter


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
        emitter = StopAwareEmitter(stop)
        finish = "length"

        for index in range(max_tokens):
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            text = f"tok{index} "

            emit, stopped = emitter.add(text)
            if emit:
                yield GenerationChunk(text=emit, finish_reason=None)
            if stopped:
                finish = "stop"
                break

        if finish == "length":
            remaining = emitter.flush()
            if remaining:
                yield GenerationChunk(text=remaining, finish_reason=None)

        yield GenerationChunk(text="", finish_reason=finish)
