import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

from cbserver.engine.protocols import FinishReason, GenerationChunk


def _default_streamer_factory(tokenizer: Any) -> Any:
    from transformers import TextIteratorStreamer

    return TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )


class HFBaselineGenerator:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        streamer_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._streamer_factory = streamer_factory or _default_streamer_factory

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        streamer = self._streamer_factory(self._tokenizer)

        do_sample = temperature > 0
        generation_kwargs: dict[str, Any] = {
            **dict(inputs),
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "pad_token_id": getattr(self._tokenizer, "pad_token_id", None),
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        thread = threading.Thread(
            target=self._model.generate, kwargs=generation_kwargs, daemon=True
        )
        thread.start()

        loop = asyncio.get_running_loop()
        accumulated = ""
        finish: FinishReason = "length"

        try:
            while True:
                chunk_text = await loop.run_in_executor(None, _next_or_none, streamer)
                if chunk_text is None:
                    break

                accumulated += chunk_text

                if stop and any(needle in accumulated for needle in stop):
                    finish = "stop"
                    yield GenerationChunk(text=chunk_text, finish_reason=None)
                    break

                yield GenerationChunk(text=chunk_text, finish_reason=None)

            yield GenerationChunk(text="", finish_reason=finish)
        finally:
            await loop.run_in_executor(None, thread.join)


def _next_or_none(iterator: Any) -> str | None:
    try:
        return next(iter(iterator) if not hasattr(iterator, "__next__") else iterator)
    except StopIteration:
        return None
