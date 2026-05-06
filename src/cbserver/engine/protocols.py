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
        stop: list[str] | None = None,
    ) -> AsyncIterator[GenerationChunk]: ...


def _earliest_stop_index(buffer: str, stop: list[str]) -> int | None:
    indices = [buffer.find(needle) for needle in stop]
    hits = [idx for idx in indices if idx >= 0]
    return min(hits) if hits else None


class StopAwareEmitter:
    """Buffers text so stop sequences spanning chunk boundaries are detected
    before the prefix of the stop sequence has been emitted to the client.

    A stop string of length L can begin in any of the previous L-1 characters,
    so we never emit the trailing L-1 characters of the buffer until either
    new text confirms they are not a stop prefix or the stream completes.
    """

    def __init__(self, stop: list[str] | None) -> None:
        self._stop = stop
        max_len = max((len(s) for s in stop), default=0) if stop else 0
        self._hold_back = max(0, max_len - 1)
        self._accumulated = ""
        self._emitted_len = 0

    def add(self, chunk_text: str) -> tuple[str | None, bool]:
        """Append text and return (text_to_emit, stopped).

        text_to_emit is None when nothing is safe to emit yet (held back
        pending stop-prefix disambiguation). stopped is True when a full
        stop sequence has been observed in the buffer; the caller should
        terminate the stream with finish_reason="stop" after yielding any
        returned text.
        """
        self._accumulated += chunk_text

        if self._stop:
            stop_idx = _earliest_stop_index(self._accumulated, self._stop)
            if stop_idx is not None:
                pre_stop_end = max(stop_idx, self._emitted_len)
                emit = self._accumulated[self._emitted_len : pre_stop_end]
                self._emitted_len = pre_stop_end
                return (emit or None), True

        safe_end = max(self._emitted_len, len(self._accumulated) - self._hold_back)
        if safe_end > self._emitted_len:
            emit = self._accumulated[self._emitted_len : safe_end]
            self._emitted_len = safe_end
            return emit, False

        return None, False

    def flush(self) -> str | None:
        """Emit any held-back text. Call when generation ends without a stop match."""
        if self._emitted_len < len(self._accumulated):
            emit = self._accumulated[self._emitted_len :]
            self._emitted_len = len(self._accumulated)
            return emit
        return None
