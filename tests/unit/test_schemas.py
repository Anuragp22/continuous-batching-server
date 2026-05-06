import pytest
from pydantic import ValidationError

from cbserver.api.schemas import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    Usage,
)


def test_request_accepts_minimal_payload() -> None:
    req = CompletionRequest(model="qwen", prompt="hello")
    assert req.max_tokens == 64
    assert req.temperature == 1.0
    assert req.stream is False


def test_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(model="qwen", prompt="")


def test_request_rejects_zero_max_tokens() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(model="qwen", prompt="hi", max_tokens=0)


def test_request_rejects_top_p_zero() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(model="qwen", prompt="hi", top_p=0.0)


def test_response_default_id_and_created() -> None:
    resp = CompletionResponse(
        model="qwen",
        choices=[Choice(text="hello", finish_reason="stop")],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    assert resp.id.startswith("cmpl-")
    assert resp.object == "text_completion"
    assert resp.created > 0


def test_stream_chunk_object_field() -> None:
    chunk = StreamChunk(
        id="cmpl-x",
        created=1,
        model="qwen",
        choices=[Choice(text="hi", finish_reason=None)],
    )
    assert chunk.object == "text_completion"
