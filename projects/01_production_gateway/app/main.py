from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import openai
from openai import AsyncOpenAI

from app.config import get_settings
from app.routers import extraction, health, stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.openai_client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )
    yield
    await app.state.openai_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    @app.exception_handler(openai.RateLimitError)
    async def rate_limit_handler(request: Request, exc: openai.RateLimitError):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "UPSTREAM_RATE_LIMIT", "detail": str(exc)},
            headers={"Retry-After": "10"},
        )

    @app.exception_handler(openai.APIConnectionError)
    async def connection_handler(request: Request, exc: openai.APIConnectionError):
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": "UPSTREAM_CONNECTION_ERROR", "detail": str(exc)},
        )

    app.include_router(health.router)
    app.include_router(extraction.router)
    app.include_router(stream.router)

    return app


app = create_app()