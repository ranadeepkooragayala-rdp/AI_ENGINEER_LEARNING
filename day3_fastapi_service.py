"""
Day 3 Project Deliverable: Production-Grade AI Inference Gateway Service
File: day3_fastapi_service.py

Key Capabilities:
- Global lifespan resource pooling (httpx.AsyncClient)
- Strict schema enforcement & response sanitization via Pydantic v2
- Modular dependency injection for Header authentication
- Route prioritization (Static routes declared before Dynamic routes)
- Asynchronous telemetry logging via BackgroundTasks
- Real-time token streaming over Server-Sent Events (SSE)

Execution:
    uvicorn day3_fastapi_service:app --reload --port 8000
"""

import asyncio
from contextlib import asynccontextmanager
from enum import Enum
from typing import Dict, List
import httpx
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ─── 1. GLOBAL STATE & LIFESPAN CONTEXT MANAGER ──────────────────────────────
# Global state dictionary used to retain long-lived resources
state: Dict[str, httpx.AsyncClient] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of application resources.
    Everything before 'yield' executes at application startup.
    Everything after 'yield' executes at application shutdown.
    """
    # STARTUP: Initialize a single persistent connection pool
    state["http_client"] = httpx.AsyncClient(timeout=10.0)
    print("🚀 [Lifespan] Connection pool initialized.")
    
    yield
    
    # SHUTDOWN: Gracefully drain and terminate the connection pool
    await state["http_client"].aclose()
    print("🛑 [Lifespan] Connection pool closed.")


app = FastAPI(
    title="AI Model Gateway Service",
    description="Production-grade AI inference routing and streaming microservice.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── 2. ENUMS & DATA CONTRACTS (PYDANTIC SCHEMAS) ───────────────────────────
class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1000, description="Input text prompt for model")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Max generated tokens")
    tags: List[str] = Field(default_factory=list, description="Categorization metadata tags")


class InferenceResponse(BaseModel):
    model_name: str
    provider: ModelProvider
    prompt: str
    completion: str
    tokens_used: int
    status: str


# ─── 3. DEPENDENCIES & WORKERS ───────────────────────────────────────────────
def get_http_client() -> httpx.AsyncClient:
    """Dependency: Injects the warm httpx client instance from state."""
    return state["http_client"]


def verify_api_key(x_api_key: str = Header(..., description="API Access Key Header")) -> str:
    """Dependency: Enforces header-based authentication on protected routes."""
    if x_api_key != "secret-ai-token-2026":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key."
        )
    return x_api_key


async def log_telemetry(prompt: str, tokens: int, client: httpx.AsyncClient):
    """Background Task: Non-blocking telemetry and metric emission."""
    await asyncio.sleep(0.5)  # Simulate I/O latency to analytics backend
    print(f"📊 [Telemetry] Successfully logged prompt ({len(prompt)} chars) | Tokens: {tokens}")


# ─── 4. STATIC & DYNAMIC ROUTES ─────────────────────────────────────────────
@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check():
    """Health check route returning ASGI runtime status."""
    return {"status": "HEALTHY", "framework": "FastAPI", "mode": "ASGI"}


@app.get("/gateway/upstream-status", status_code=status.HTTP_200_OK, tags=["System"])
async def check_upstream(client: httpx.AsyncClient = Depends(get_http_client)):
    """Verifies external network connectivity using the pooled HTTP client."""
    response = await client.get("https://httpbin.org/status/200")
    return {"upstream_status_code": response.status_code}


# NOTE: Static route MUST be registered before the dynamic route below
@app.get("/models/default", status_code=status.HTTP_200_OK, tags=["Models"])
async def get_default_model():
    """Static lookup for default fallback model configuration."""
    return {
        "default_model": "llama-3-8b",
        "provider": ModelProvider.OLLAMA,
        "context_window": 8192,
    }


@app.get("/models/{model_id}", status_code=status.HTTP_200_OK, tags=["Models"])
async def get_model_info(model_id: int):
    """Dynamic lookup for specific model IDs with integer validation."""
    if model_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model ID must be a positive integer.",
        )
    return {"model_id": model_id, "status": "AVAILABLE"}


# ─── 5. CORE INFERENCE & STREAMING ENDPOINTS ─────────────────────────────────
@app.post(
    "/models/{provider}/generate",
    response_model=InferenceResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Inference"]
)
async def generate_completion(
    provider: ModelProvider,
    payload: InferenceRequest,
    background_tasks: BackgroundTasks,
    dry_run: bool = False,
    auth_token: str = Depends(verify_api_key),
    client: httpx.AsyncClient = Depends(get_http_client),
):
    """
    Primary inference endpoint handling validation, mock generation,
    background metric logging, and response sanitization.
    """
    if dry_run:
        return {
            "model_name": f"{provider.value}-mock",
            "provider": provider,
            "prompt": payload.prompt,
            "completion": "[DRY RUN] No inference executed.",
            "tokens_used": 0,
            "status": "VALIDATED",
            "internal_debug_secret": "STRIPPED_BY_RESPONSE_MODEL"
        }

    generated_text = f"Simulated output from {provider.value} for: '{payload.prompt}'"
    tokens_count = len(payload.prompt.split()) + 10

    # Dispatch non-blocking telemetry task
    background_tasks.add_task(log_telemetry, payload.prompt, tokens_count, client)

    return {
        "model_name": f"{provider.value}-prod-v1",
        "provider": provider,
        "prompt": payload.prompt,
        "completion": generated_text,
        "tokens_used": tokens_count,
        "status": "COMPLETED",
        "internal_debug_secret": "STRIPPED_BY_RESPONSE_MODEL"
    }


async def stream_tokens(prompt: str):
    """Async generator yielding chunks formatted for Server-Sent Events (SSE)."""
    words = prompt.split()
    for word in words:
        await asyncio.sleep(0.15)  # Simulate per-token generation latency
        yield f"data: {word}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/stream/inference", tags=["Inference"])
async def stream_inference(
    prompt: str = "FastAPI streaming is responsive",
    auth_token: str = Depends(verify_api_key),
):
    """Streams token chunks in real-time using text/event-stream media type."""
    return StreamingResponse(
        stream_tokens(prompt),
        media_type="text/event-stream",
    )
