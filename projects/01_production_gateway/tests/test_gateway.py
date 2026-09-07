import json
import pytest
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_health_check(async_client):
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"


@pytest.mark.asyncio
async def test_extraction_success(async_client, mock_openai_client):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "server_name": "prod-api-01",
        "cpu_usage": 45.2,
        "memory_usage": 68.0,
        "status": "HEALTHY",
    })
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_completion

    response = await async_client.post(
        "/v1/extract",
        json={"raw_log": "[SYSTEM] prod-api-01 load is standard. cpu at 45.2%, mem 68.0%"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["server_name"] == "prod-api-01"
    assert body["status"] == "HEALTHY"


@pytest.mark.asyncio
async def test_streaming_sse_format(async_client, mock_openai_client):
    chunk1 = MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))], usage=None)
    chunk2 = MagicMock(choices=[MagicMock(delta=MagicMock(content=" World"))], usage=None)
    chunk3 = MagicMock(
        choices=[],
        usage=MagicMock(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )

    async def mock_stream_iterator():
        for c in [chunk1, chunk2, chunk3]:
            yield c

    mock_openai_client.chat.completions.create.return_value = mock_stream_iterator()

    response = await async_client.post(
        "/v1/chat/stream",
        json={"prompt": "Say hello"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers["x-accel-buffering"] == "no"
    assert "data: [DONE]\n\n" in response.text