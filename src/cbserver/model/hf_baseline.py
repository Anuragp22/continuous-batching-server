import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

from cbserver.engine.protocols import FinishReason, GenerationChunk, StopAwareEmitter

logger = logging.getLogger(__name__)


def _default_streamer_factory(tokenizer: Any) -> Any:
    from transformers import TextIteratorStreamer

    return TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )


def _build_cancel_criteria(cancel_flag: threading.Event) -> Any | None:
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return None

    class _FlagStoppingCriterion(StoppingCriteria):  # type: ignore[misc]
        def __call__(self, input_ids: Any, scores: Any, **_kwargs: Any) -> bool:
            return cancel_flag.is_set()

    return StoppingCriteriaList([_FlagStoppingCriterion()])


def _terminate_streamer(streamer: Any) -> None:
    end = getattr(streamer, "end", None) or getattr(streamer, "put_stop", None)
    if end is None:
        return
    try:
        end()
    except Exception:  # pragma: no cover - best-effort cleanup
        logger.exception("hf_baseline: failed to terminate streamer")


class HFBaselineGenerator:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        streamer_factory: Callable[[Any], Any] | None = None,
        cancel_criteria_factory: Callable[[threading.Event], Any] | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._streamer_factory = streamer_factory or _default_streamer_factory
        self._cancel_criteria_factory = cancel_criteria_factory or _build_cancel_criteria

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
        cancel_flag = threading.Event()
        thread_exception: list[BaseException | None] = [None]

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

        criteria = self._cancel_criteria_factory(cancel_flag)
        if criteria is not None:
            generation_kwargs["stopping_criteria"] = criteria

        def _run_generate() -> None:
            try:
                self._model.generate(**generation_kwargs)
            except BaseException as exc:
                logger.exception("hf_baseline: model.generate raised")
                thread_exception[0] = exc
            finally:
                _terminate_streamer(streamer)

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        loop = asyncio.get_running_loop()
        emitter = StopAwareEmitter(stop)
        finish: FinishReason = "length"

        try:
            while True:
                chunk_text = await loop.run_in_executor(None, _next_or_none, streamer)
                if chunk_text is None:
                    break

                emit, stopped = emitter.add(chunk_text)
                if emit:
                    yield GenerationChunk(text=emit, finish_reason=None)
                if stopped:
                    finish = "stop"
                    break

            if thread_exception[0] is not None:
                raise thread_exception[0]

            if finish == "length":
                remaining = emitter.flush()
                if remaining:
                    yield GenerationChunk(text=remaining, finish_reason=None)

            yield GenerationChunk(text="", finish_reason=finish)
        finally:
            cancel_flag.set()


def _next_or_none(iterator: Any) -> str | None:
    try:
        return next(iter(iterator) if not hasattr(iterator, "__next__") else iterator)
    except StopIteration:
        return None
