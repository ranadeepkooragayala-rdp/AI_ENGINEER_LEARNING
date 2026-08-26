"""
Day 5 Project Deliverable: Error Handling, Rate Limiting, and Self-Correction Gateway

To run:
    uvicorn day5_resilient_gateway:app --reload --port 8000
"""

import os
import time
import json
import re
import random
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

# ─── 1. GLOBAL LIFESPAN & STATE ──────────────────────────────────────────────
state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Connection pooling
    state["openai_client"] = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "mock-openai-key"),
        timeout=20.0,
        max_retries=2
    )
    print("[Lifespan] Resilient AI Gateway started.")
    yield
    # SHUTDOWN
    await state["openai_client"].close()
    print("[Lifespan] Connection pools closed.")

app = FastAPI(
    title="Day 5 Resilient AI Gateway",
    version="1.0.0",
    lifespan=lifespan
)

def get_openai_client() -> AsyncOpenAI:
    return state["openai_client"]

# ─── 2. GLOBAL EXCEPTION HANDLERS ───────────────────────────────────
@app.exception_handler(openai.RateLimitError)
async def openai_rate_limit_handler(request: Request, exc: openai.RateLimitError):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "UPSTREAM_RATE_LIMIT",
            "message": "Upstream AI provider rate limit was exceeded. Retry after delay.",
            "details": str(exc),
        },
        headers={"Retry-After": "10"}
    )

@app.exception_handler(openai.APITimeoutError)
async def openai_timeout_handler(request: Request, exc: openai.APITimeoutError):
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={
            "error": "UPSTREAM_TIMEOUT",
            "message": "AI inference timed out. Please reduce context size or retry.",
        }
    )

@app.exception_handler(openai.APIConnectionError)
async def openai_connection_handler(request: Request, exc: openai.APIConnectionError):
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": "UPSTREAM_CONNECTION_FAILED",
            "message": "Failed to connect to upstream AI provider network.",
        }
    )

@app.exception_handler(openai.BadRequestError)
async def openai_bad_request_handler(request: Request, exc: openai.BadRequestError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "INVALID_PROMPT_OR_PAYLOAD",
            "detail": str(exc),
        }
    )

# ─── 3. SLIDING WINDOW RATE LIMITING  ───────────────────────────────
REQUEST_HISTORY: dict[str, list[float]] = defaultdict(list)

def rate_limiter(max_requests: int = 5, window_seconds: int = 60):
    """Enforces in-memory sliding window rate limits per client IP."""
    def dependency(request: Request):
        client_ip = request.client.host if request.client else "anonymous"
        now = time.time()

        # Evict timestamps outside current sliding window
        REQUEST_HISTORY[client_ip] = [
            t for t in REQUEST_HISTORY[client_ip] if now - t < window_seconds
        ]

        if len(REQUEST_HISTORY[client_ip]) >= max_requests:
            oldest_request = REQUEST_HISTORY[client_ip][0]
            retry_after = int(window_seconds - (now - oldest_request)) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Client rate limit exceeded. Please back off.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0"
                }
            )

        REQUEST_HISTORY[client_ip].append(now)
        return True

    return dependency

# ─── 4. DATA SCHEMAS ─────────────────────────────────────────────────────────
class ServerMetricReport(BaseModel):
    server_name: str = Field(description="Hostname or instance identifier")
    cpu_usage: float = Field(description="Percentage CPU utilization", ge=0.0, le=100.0)
    memory_usage: float = Field(description="Percentage RAM utilization", ge=0.0, le=100.0)
    status: Literal["HEALTHY", "WARNING", "CRITICAL"] = Field(description="Operational status")

class ServerLogRequest(BaseModel):
    log_text: str = Field(..., min_length=15, description="Raw server telemetry log")

# ─── 5. DEFENSIVE PARSER & SELF-CORRECTION ────────────────────
def robust_json_parser(raw_response: str, model_cls: type[BaseModel]) -> BaseModel:
    cleaned = raw_response.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        parsed_dict = json.loads(cleaned)
        return model_cls.model_validate(parsed_dict)
    except (json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"Schema validation failed for {model_cls.__name__}: {error}")

async def call_with_backoff(coro_fn, *args, max_retries: int = 2, base_delay: float = 1.0, **kwargs):
    """Wraps async calls with exponential backoff and jitter."""
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as exc:
            if attempt == max_retries:
                raise exc
            delay = random.uniform(0, base_delay * (2 ** attempt))
            print(f"[Backoff] Transient error ({type(exc).__name__}). Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)

async def extract_with_self_correction(
    raw_log: str,
    client: AsyncOpenAI,
    max_retries: int = 2
) -> ServerMetricReport:
    """Self-correcting feedback loop that feeds validation errors back to the LLM."""
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an SRE telemetry parser. Extract metrics into a valid JSON object matching: "
                "server_name (str), cpu_usage (0.0-100.0), memory_usage (0.0-100.0), status ('HEALTHY'|'WARNING'|'CRITICAL'). "
                "Return ONLY valid JSON."
            )
        },
        {"role": "user", "content": raw_log}
    ]

    for attempt in range(max_retries + 1):
        completion = await call_with_backoff(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0
        )
        content = completion.choices[0].message.content or ""

        try:
            return robust_json_parser(content, ServerMetricReport)
        except (ValueError, ValidationError) as err:
            if attempt == max_retries:
                raise ValueError(f"Failed self-correction after {max_retries} retries. Error: {err}")

            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Your output failed validation: {str(err)}. Correct the JSON and return valid output."
            })

# ─── 6. ENDPOINTS ────────────────────────────────────────────────────────────
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "HEALTHY", "gateway": "Resilient AI Gateway"}

@app.post(
    "/metrics/extract",
    response_model=ServerMetricReport,
    status_code=status.HTTP_200_OK
)
async def extract_metrics_endpoint(
    payload: ServerLogRequest,
    client: AsyncOpenAI = Depends(get_openai_client),
    _rate_limit: bool = Depends(rate_limiter(max_requests=5, window_seconds=60))
):
    try:
        report = await extract_with_self_correction(payload.log_text, client)
        return report
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream inference error: {str(err)}"
        )
