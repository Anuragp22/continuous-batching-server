import pytest

from cbserver.engine.mock import MockGenerator


def _content(chunks: list) -> str:
    return "".join(c.text for c in chunks)


@pytest.mark.asyncio
async def test_mock_yields_max_tokens_plus_terminator() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=4)]
    assert _content(chunks) == "tok0 tok1 tok2 tok3 "
    assert chunks[-1].finish_reason == "length"


@pytest.mark.asyncio
async def test_mock_chunks_have_text() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=3)]
    assert _content(chunks) == "tok0 tok1 tok2 "


@pytest.mark.asyncio
async def test_mock_stops_when_single_stop_sequence_matches() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=10, stop=["tok2"])]
    assert _content(chunks) == "tok0 tok1 "
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_stops_with_first_matching_sequence() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [
        chunk async for chunk in gen.stream("hi", max_tokens=10, stop=["never", "tok1"])
    ]
    assert _content(chunks) == "tok0 "
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_ignores_non_matching_stop() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [
        chunk async for chunk in gen.stream("hi", max_tokens=3, stop=["nonexistent"])
    ]
    assert _content(chunks) == "tok0 tok1 tok2 "
    assert chunks[-1].finish_reason == "length"


@pytest.mark.asyncio
async def test_mock_with_none_stop_behaves_like_no_stop() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=3, stop=None)]
    assert len(chunks) == 4
    assert chunks[-1].finish_reason == "length"


@pytest.mark.asyncio
async def test_mock_picks_earliest_stop_index_not_list_order() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=10, stop=["k2", "k1"])]
    assert _content(chunks) == "tok0 to"
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_emits_pre_stop_portion_when_stop_matches_inside_chunk() -> None:
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=10, stop=["k1"])]
    assert _content(chunks) == "tok0 to"
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_catches_stop_straddling_chunk_boundary() -> None:
    """Stop sequence ``"0 t"`` straddles ``"tok0 "`` and ``"tok1 "``.
    The end of the first chunk would have leaked the prefix of the stop
    if the buffered emitter didn't hold back trailing characters."""
    gen = MockGenerator(delay_per_token=0.0)
    chunks = [chunk async for chunk in gen.stream("hi", max_tokens=10, stop=["0 t"])]
    assert _content(chunks) == "tok"
    assert chunks[-1].finish_reason == "stop"
