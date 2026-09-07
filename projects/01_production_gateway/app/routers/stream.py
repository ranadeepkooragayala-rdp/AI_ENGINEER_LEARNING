from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.dependencies import get_llm_client
from app.schemas import StreamRequest
from app.services.streamer import sse_token_generator

router = APIRouter(prefix="/v1", tags=["Streaming"])


@router.post("/chat/stream")
async def stream_endpoint(
    payload: StreamRequest,
    request: Request,
    client: AsyncOpenAI = Depends(get_llm_client),
) -> StreamingResponse:
    return StreamingResponse(
        sse_token_generator(payload.prompt, client, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )