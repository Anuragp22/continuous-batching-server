import asyncio
from collections.abc import AsyncIterator

from cbserver.engine.protocols import GenerationChunk


def _earliest_stop_index(buffer: str, stop: list[str]) -> int | None:
    indices = [buffer.find(needle) for needle in stop]
    hits = [idx for idx in indices if idx >= 0]
    return min(hits) if hits else None


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
        accumulated = ""
        for index in range(max_tokens):
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            text = f"tok{index} "
            new_accumulated = accumulated + text

            if stop:
                stop_idx = _earliest_stop_index(new_accumulated, stop)
                if stop_idx is not None:
                    pre_stop = new_accumulated[len(accumulated) : stop_idx]
                    if pre_stop:
                        yield GenerationChunk(text=pre_stop, finish_reason=None)
                    yield GenerationChunk(text="", finish_reason="stop")
                    return

            yield GenerationChunk(text=text, finish_reason=None)
            accumulated = new_accumulated

        yield GenerationChunk(text="", finish_reason="length")
