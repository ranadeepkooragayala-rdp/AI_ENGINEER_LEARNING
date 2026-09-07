# File: projects/01_production_gateway/app/dependencies.py

import time
from collections import defaultdict
from typing import Dict, List
from fastapi import HTTPException, Request, status
from openai import AsyncOpenAI

_REQUEST_HISTORY: Dict[str, List[float]] = defaultdict(list)


def get_llm_client(request: Request) -> AsyncOpenAI:
    """Injects client pool initialized in FastAPI lifespan."""
    return request.app.state.openai_client


def rate_limiter(max_requests: int = 5, window_seconds: int = 60):
    """Sliding-window in-memory IP rate limiter."""
    def dependency(request: Request):
        client_ip = request.client.host if request.client else "anonymous"
        now = time.time()

        _REQUEST_HISTORY[client_ip] = [
            t for t in _REQUEST_HISTORY[client_ip] if now - t < window_seconds
        ]

        if len(_REQUEST_HISTORY[client_ip]) >= max_requests:
            oldest = _REQUEST_HISTORY[client_ip][0]
            retry_after = int(window_seconds - (now - oldest)) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please back off.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        _REQUEST_HISTORY[client_ip].append(now)
        return True

    return dependency