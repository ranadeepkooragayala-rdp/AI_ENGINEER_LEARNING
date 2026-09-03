"""
Day 6 Project Deliverable: Real-Time SSE Token Streaming & Billing Microservice

To run:
    uvicorn day6_streaming_sse:app --reload --port 8000
"""

import os
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Pricing rates for gpt-4o-mini
INPUT_TOKEN_PRICE = 0.15 / 1_000_000
OUTPUT_TOKEN_PRICE = 0.60 / 1_000_000

# ─── 1. GLOBAL STATE & LIFESPAN MANAGEMENT ───────────────────────────────────
app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize connection pool
    app_state["openai_client"] = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "mock-openai-key"),
        timeout=30.0,
    )
    print("[Lifespan] OpenAI streaming client initialized.")
    yield
    # Clean shutdown
    await app_state["openai_client"].close()
    print("[Lifespan] Connection pool terminated.")

app = FastAPI(
    title="Day 6 SSE Token Streaming Service",
    version="1.0.0",
    lifespan=lifespan,
)

def get_llm_client() -> AsyncOpenAI:
    return app_state["openai_client"]

# ─── 2. DATA CONTRACTS ───────────────────────────────────────────────────────
class StreamRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt text to generate tokens for")

# ─── 3. ASYNC SSE GENERATOR WITH BILLING & DISCONNECT MONITORING ─────────────
async def sse_token_generator(
    prompt: str,
    client: AsyncOpenAI,
    request: Request
) -> AsyncGenerator[str, None]:
    """Streams completion chunks via SSE and computes exact usage costs."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in response:
        # Check if the client terminated the HTTP connection early
        if await request.is_disconnected():
            print("[SSE] Client disconnected early. Halting upstream token consumption.")
            break

        # 1. Stream content tokens
        if chunk.choices and chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            payload = json.dumps({"type": "token", "token": token})
            yield f"data: {payload}\n\n"

        # 2. Stream final token usage and calculated billing
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
            total_cost_usd = (
                prompt_tokens * INPUT_TOKEN_PRICE + completion_tokens * OUTPUT_TOKEN_PRICE
            )
            usage_payload = json.dumps({
                "type": "usage",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
                "cost_usd": round(total_cost_usd, 8),
            })
            yield f"data: {usage_payload}\n\n"

    # 3. Standard SSE stream completion event
    yield "data: [DONE]\n\n"

# ─── 4. STREAMING ENDPOINTS ──────────────────────────────────────────────────
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "HEALTHY", "service": "SSE Streaming Gateway"}

@app.post("/v1/chat/stream")
async def stream_chat(
    payload: StreamRequest,
    request: Request,
    client: AsyncOpenAI = Depends(get_llm_client)
) -> StreamingResponse:
    generator = sse_token_generator(payload.prompt, client, request)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
