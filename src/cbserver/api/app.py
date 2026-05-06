import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cbserver.api.routes_completions import router as completions_router
from cbserver.api.routes_health import router as health_router
from cbserver.config import Settings, get_settings
from cbserver.engine.mock import MockGenerator
from cbserver.engine.protocols import TextGenerator
from cbserver.obs.logging import configure_logging, get_logger


def _build_generator(settings: Settings, log) -> TextGenerator:
    if settings.engine_backend == "hf":
        from cbserver.model.hf_baseline import HFBaselineGenerator
        from cbserver.model.loader import load_model_and_tokenizer

        log.info("model.loading", model=settings.model)
        loaded = load_model_and_tokenizer(
            settings.model,
            device=settings.device,
            dtype=settings.dtype,
            attn_implementation=settings.attn_implementation,
        )
        log.info(
            "model.loaded",
            model=settings.model,
            device=loaded.device,
            dtype=loaded.dtype,
        )
        return HFBaselineGenerator(
            model=loaded.model, tokenizer=loaded.tokenizer, device=loaded.device
        )

    return MockGenerator(delay_per_token=settings.mock_token_delay)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, format=settings.log_format)
    log = get_logger("cbserver.api")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "server.startup",
            engine_backend=settings.engine_backend,
            model=settings.model,
            host=settings.host,
            port=settings.port,
            max_concurrent_requests=settings.max_concurrent_requests,
        )
        app.state.concurrency_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        app.state.generator = _build_generator(settings, log)
        yield
        log.info("server.shutdown")

    app = FastAPI(
        title="cbserver",
        version="0.1.0",
        description="Continuous-batching inference server",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(completions_router)
    return app
