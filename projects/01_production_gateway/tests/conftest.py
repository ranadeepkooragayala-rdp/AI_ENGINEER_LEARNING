from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.dependencies import get_llm_client


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest_asyncio.fixture
async def async_client(mock_openai_client) -> AsyncGenerator[AsyncClient, None]:
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: mock_openai_client

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client