import asyncio

import pytest

from cbserver.api.routes_completions import _stream_events
from cbserver.api.schemas import CompletionRequest
from cbserver.engine.mock import MockGenerator
from cbserver.obs.metrics import REGISTRY


class _FakeRequest:
    def __init__(self, *, disconnected: bool = False) -> None:
        self._disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self._disconnected


def _cancelled_count() -> float:
    value = REGISTRY.get_sample_value(
        "cbserver_requests_total", {"status": "cancelled"}
    )
    return value or 0.0


def _decoded_count() -> float:
    value = REGISTRY.get_sample_value("cbserver_decode_tokens_total")
    return value or 0.0


@pytest.mark.asyncio
async def test_stream_marks_cancelled_when_consumer_aclose() -> None:
    body = CompletionRequest(model="qwen", prompt="hi", max_tokens=100, stream=True)
    semaphore = asyncio.Semaphore(1)
    events = _stream_events(
        body, MockGenerator(delay_per_token=0.0), semaphore, _FakeRequest()
    )

    cancelled_before = _cancelled_count()
    decoded_before = _decoded_count()

    await events.__anext__()
    await events.__anext__()
    await events.aclose()

    assert _cancelled_count() - cancelled_before == 1
    assert _decoded_count() - decoded_before == 2


@pytest.mark.asyncio
async def test_stream_marks_cancelled_when_request_disconnects() -> None:
    body = CompletionRequest(model="qwen", prompt="hi", max_tokens=100, stream=True)
    semaphore = asyncio.Semaphore(1)
    events = _stream_events(
        body,
        MockGenerator(delay_per_token=0.0),
        semaphore,
        _FakeRequest(disconnected=True),
    )

    cancelled_before = _cancelled_count()

    chunks_seen = 0
    async for _ in events:
        chunks_seen += 1
        if chunks_seen > 5:
            break

    assert _cancelled_count() - cancelled_before == 1
    assert chunks_seen == 0


@pytest.mark.asyncio
async def test_stream_releases_semaphore_after_cancellation() -> None:
    body = CompletionRequest(model="qwen", prompt="hi", max_tokens=100, stream=True)
    semaphore = asyncio.Semaphore(1)
    events = _stream_events(
        body, MockGenerator(delay_per_token=0.0), semaphore, _FakeRequest()
    )

    await events.__anext__()
    assert semaphore.locked()
    await events.aclose()
    assert not semaphore.locked()
