import pytest

from cbserver.engine.mock import MockGenerator


@pytest.mark.asyncio
async def test_mock_yields_max_tokens_plus_terminator() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=4)]
    assert len(chunks) == 5
    assert all(c.finish_reason is None for c in chunks[:-1])
    assert chunks[-1].text == ""
    assert chunks[-1].finish_reason == "length"


@pytest.mark.asyncio
async def test_mock_chunks_have_text() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=3)]
    assert chunks[0].text == "tok0 "
    assert chunks[1].text == "tok1 "
    assert chunks[2].text == "tok2 "


@pytest.mark.asyncio
async def test_mock_stops_when_single_stop_sequence_matches() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=10, stop=["tok2"])]
    texts = [c.text for c in chunks if c.text]
    assert texts == ["tok0 ", "tok1 "]
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_stops_with_first_matching_sequence() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [
        chunk async for chunk in gen.stream("hi", max_tokens=10, stop=["never", "tok1"])
    ]
    texts = [c.text for c in chunks if c.text]
    assert texts == ["tok0 "]
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_ignores_non_matching_stop() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [
        chunk async for chunk in gen.stream("hi", max_tokens=3, stop=["nonexistent"])
    ]
    texts = [c.text for c in chunks if c.text]
    assert texts == ["tok0 ", "tok1 ", "tok2 "]
    assert chunks[-1].finish_reason == "length"


@pytest.mark.asyncio
async def test_mock_with_none_stop_behaves_like_no_stop() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=3, stop=None)]
    assert len(chunks) == 4
    assert chunks[-1].finish_reason == "length"
