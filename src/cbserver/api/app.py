import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cbserver.api.routes_completions import router as completions_router
from cbserver.api.routes_health import router as health_router
from cbserver.config import get_settings
from cbserver.engine.mock import MockGenerator
from cbserver.obs.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, format=settings.log_format)
    log = get_logger("cbserver.api")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "server.startup",
            model=settings.model,
            host=settings.host,
            port=settings.port,
            max_concurrent_requests=settings.max_concurrent_requests,
        )
        app.state.concurrency_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        app.state.generator = MockGenerator(delay_per_token=settings.mock_token_delay)
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
