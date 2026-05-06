import queue
import threading
from typing import Any

import pytest

from cbserver.model.hf_baseline import HFBaselineGenerator


class _FakeBatchEncoding(dict):
    def to(self, _device: str) -> "_FakeBatchEncoding":
        return self


class _FakeTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    def __call__(self, _text: str, return_tensors: str | None = None) -> _FakeBatchEncoding:
        return _FakeBatchEncoding(input_ids=[[1, 2, 3]])


_STOP_SIGNAL = object()


class _FakeStreamer:
    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()

    def put(self, value: Any) -> None:
        self._queue.put(value)

    def put_stop(self) -> None:
        self._queue.put(_STOP_SIGNAL)

    def __iter__(self) -> "_FakeStreamer":
        return self

    def __next__(self) -> str:
        item = self._queue.get()
        if item is _STOP_SIGNAL:
            raise StopIteration
        return item


class _FakeModel:
    def __init__(self, output_chunks: list[str], delay: float = 0.0) -> None:
        self._chunks = output_chunks
        self._delay = delay

    def generate(self, *, streamer: _FakeStreamer, **_kwargs: Any) -> None:
        import time

        for chunk in self._chunks:
            if self._delay > 0:
                time.sleep(self._delay)
            streamer.put(chunk)
        streamer.put_stop()


def _make_generator(output_chunks: list[str]) -> tuple[HFBaselineGenerator, _FakeStreamer]:
    streamer = _FakeStreamer()
    model = _FakeModel(output_chunks)
    tokenizer = _FakeTokenizer()
    gen = HFBaselineGenerator(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        streamer_factory=lambda _tok: streamer,
    )
    return gen, streamer


@pytest.mark.asyncio
async def test_yields_text_chunks_then_length_terminator() -> None:
    gen, _ = _make_generator(["hello", " world", "!"])
    chunks = [chunk async for chunk in gen.stream(prompt="hi", max_tokens=10)]

    text_chunks = [c for c in chunks if c.text]
    assert [c.text for c in text_chunks] == ["hello", " world", "!"]
    assert chunks[-1].text == ""
    assert chunks[-1].finish_reason == "length"


@pytest.mark.asyncio
async def test_stop_sequence_does_not_echo_in_output() -> None:
    gen, _ = _make_generator(["hello ", "world ", "and more"])
    chunks = [
        chunk async for chunk in gen.stream(prompt="hi", max_tokens=10, stop=["world"])
    ]

    text_chunks = [c.text for c in chunks if c.text]
    assert text_chunks == ["hello "]
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_stop_in_middle_of_chunk_emits_pre_stop_portion_only() -> None:
    gen, _ = _make_generator(["hel", "lo END world"])
    chunks = [
        chunk async for chunk in gen.stream(prompt="hi", max_tokens=10, stop=["END"])
    ]

    text_chunks = [c.text for c in chunks if c.text]
    assert text_chunks == ["hel", "lo "]
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_stop_with_no_match_runs_to_completion() -> None:
    gen, _ = _make_generator(["a ", "b ", "c"])
    chunks = [
        chunk async for chunk in gen.stream(prompt="hi", max_tokens=10, stop=["xyz"])
    ]
    text_chunks = [c.text for c in chunks if c.text]
    assert text_chunks == ["a ", "b ", "c"]
    assert chunks[-1].finish_reason == "length"


@pytest.mark.asyncio
async def test_aclose_terminates_generator_cleanly() -> None:
    gen, streamer = _make_generator(["a", "b", "c", "d", "e"])
    iterator = gen.stream(prompt="hi", max_tokens=10)

    first = await iterator.__anext__()
    assert first.text == "a"
    await iterator.aclose()


@pytest.mark.asyncio
async def test_aclose_signals_cancel_flag_to_stopping_criteria() -> None:
    streamer = _FakeStreamer()
    cancel_observed = threading.Event()

    def _capture_factory(cancel_flag: threading.Event) -> list[Any]:
        def _check(*_args: Any, **_kwargs: Any) -> bool:
            if cancel_flag.is_set():
                cancel_observed.set()
                return True
            return False

        return [_check]

    class _LongRunningModel:
        def generate(
            self,
            *,
            streamer: _FakeStreamer,
            stopping_criteria: list[Any] | None = None,
            **_kwargs: Any,
        ) -> None:
            import time

            i = 0
            while True:
                if stopping_criteria is not None:
                    for criterion in stopping_criteria:
                        if criterion(None, None):
                            streamer.put_stop()
                            return
                streamer.put(f"tok{i} ")
                i += 1
                time.sleep(0.005)
                if i > 200:
                    streamer.put_stop()
                    return

    gen = HFBaselineGenerator(
        model=_LongRunningModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        streamer_factory=lambda _tok: streamer,
        cancel_criteria_factory=_capture_factory,
    )

    iterator = gen.stream(prompt="hi", max_tokens=10_000)
    first = await iterator.__anext__()
    assert first.text == "tok0 "
    await iterator.aclose()

    assert cancel_observed.wait(timeout=1.0), (
        "cancel_flag was not propagated to stopping_criteria; "
        "thread would have run to max_tokens and held resources"
    )


def test_generation_runs_in_separate_thread() -> None:
    """The generate() call must be threaded so the event loop can yield between chunks."""
    streamer = _FakeStreamer()

    main_thread_id = threading.get_ident()
    captured_thread_id: list[int] = []

    class _IntrospectiveModel:
        def generate(self, *, streamer: _FakeStreamer, **_kwargs: Any) -> None:
            captured_thread_id.append(threading.get_ident())
            streamer.put("ok")
            streamer.put_stop()

    gen = HFBaselineGenerator(
        model=_IntrospectiveModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        streamer_factory=lambda _tok: streamer,
    )

    import asyncio

    async def _run() -> None:
        chunks = [c async for c in gen.stream(prompt="hi", max_tokens=4)]
        assert any(c.text == "ok" for c in chunks)

    asyncio.run(_run())
    assert captured_thread_id and captured_thread_id[0] != main_thread_id
