import json
from typing import AsyncGenerator
from fastapi import Request
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()


async def sse_token_generator(
    prompt: str,
    client: AsyncOpenAI,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Streams completion chunks via SSE and computes real-time token billing."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in response:
        if await request.is_disconnected():
            break

        if chunk.choices and chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
            total_cost = (
                prompt_tokens * settings.input_token_price
                + completion_tokens * settings.output_token_price
            )
            usage_payload = json.dumps({
                "type": "usage",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
                "cost_usd": round(total_cost, 8),
            })
            yield f"data: {usage_payload}\n\n"

    yield "data: [DONE]\n\n"