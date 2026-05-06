import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from cbserver.api.schemas import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    Usage,
    _new_completion_id,
    _now,
)
from cbserver.engine.protocols import FinishReason, TextGenerator
from cbserver.obs.metrics import (
    active_requests,
    decode_tokens_total,
    requests_total,
)

router = APIRouter()


def get_generator(request: Request) -> TextGenerator:
    return request.app.state.generator


def get_semaphore(request: Request) -> asyncio.Semaphore:
    return request.app.state.concurrency_semaphore


@router.post("/v1/completions", response_model=None)
async def create_completion(
    body: CompletionRequest,
    request: Request,
    generator: TextGenerator = Depends(get_generator),
    semaphore: asyncio.Semaphore = Depends(get_semaphore),
) -> JSONResponse | EventSourceResponse:
    if body.stream:
        return EventSourceResponse(
            _stream_events(body, generator, semaphore, request),
            ping=15,
        )
    response = await _collect_response(body, generator, semaphore, request)
    return JSONResponse(content=response.model_dump())


async def _stream_events(
    body: CompletionRequest,
    generator: TextGenerator,
    semaphore: asyncio.Semaphore,
    request: Request,
) -> AsyncIterator[dict[str, str]]:
    completion_id = _new_completion_id()
    created = _now()
    decoded = 0
    cancelled = False

    async with semaphore:
        active_requests.inc()
        try:
            async for chunk in generator.stream(
                prompt=body.prompt,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                stop=body.stop,
            ):
                if await request.is_disconnected():
                    cancelled = True
                    return

                if chunk.text:
                    decoded += 1

                stream_chunk = StreamChunk(
                    id=completion_id,
                    created=created,
                    model=body.model,
                    choices=[Choice(text=chunk.text, finish_reason=chunk.finish_reason)],
                )
                yield {"data": stream_chunk.model_dump_json()}

            requests_total.labels(status="completed").inc()
            yield {"data": "[DONE]"}
        except (asyncio.CancelledError, GeneratorExit):
            cancelled = True
            raise
        finally:
            decode_tokens_total.inc(decoded)
            if cancelled:
                requests_total.labels(status="cancelled").inc()
            active_requests.dec()


async def _collect_response(
    body: CompletionRequest,
    generator: TextGenerator,
    semaphore: asyncio.Semaphore,
    request: Request,
) -> CompletionResponse:
    pieces: list[str] = []
    finish: FinishReason = "stop"
    decoded = 0
    cancelled = False

    async with semaphore:
        active_requests.inc()
        try:
            async for chunk in generator.stream(
                prompt=body.prompt,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                stop=body.stop,
            ):
                if chunk.text:
                    pieces.append(chunk.text)
                    decoded += 1
                if chunk.finish_reason:
                    finish = chunk.finish_reason

            requests_total.labels(status="completed").inc()
            prompt_tokens = _estimate_tokens(body.prompt)
            return CompletionResponse(
                model=body.model,
                choices=[Choice(text="".join(pieces), finish_reason=finish)],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=decoded,
                    total_tokens=prompt_tokens + decoded,
                ),
            )
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            decode_tokens_total.inc(decoded)
            if cancelled:
                requests_total.labels(status="cancelled").inc()
            active_requests.dec()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))
