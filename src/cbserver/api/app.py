from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from cbserver.api.routes_health import router as health_router
from cbserver.config import get_settings
from cbserver.obs.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, format=settings.log_format)
    log = get_logger("cbserver.api")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log.info("server.startup", model=settings.model, host=settings.host, port=settings.port)
        yield
        log.info("server.shutdown")

    app = FastAPI(
        title="cbserver",
        version="0.1.0",
        description="Continuous-batching inference server",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    return app
