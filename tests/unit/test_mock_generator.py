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
