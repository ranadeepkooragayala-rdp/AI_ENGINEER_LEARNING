"""
Day 4 Project Deliverable: Direct Model Integration with Structured JSON Extraction

To run:
    uvicorn day4_structured_extractor:app --reload --port 8000
"""

import os
import json
import re
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List
from fastapi import Depends, FastAPI, HTTPException, status
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

# ─── 1. GLOBAL STATE & LIFESPAN MANAGEMENT ───────────────────────────────────
state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Initialize both connection pools into global state
    state["openai_client"] = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "mock-openai-key"),
        timeout=20.0,
        max_retries=2,
    )
    state["httpx_client"] = httpx.AsyncClient(timeout=15.0)
    print(" [Lifespan] OpenAI and HTTPX clients initialized.")
    
    yield
    
    # SHUTDOWN: Gracefully close both clients
    await state["openai_client"].close()
    await state["httpx_client"].aclose()
    print(" [Lifespan] Both clients safely terminated.")

app = FastAPI(
    title="Day 4 Structured Extraction Service",
    version="1.0.0",
    lifespan=lifespan,
)

# Dependency Functions
def get_openai_client() -> AsyncOpenAI:
    return state["openai_client"]

def get_httpx_client() -> httpx.AsyncClient:
    return state["httpx_client"]

# ─── 2. DATA CONTRACTS & SCHEMAS ─────────────────────────────────────────────
class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class RootCause(BaseModel):
    component: str = Field(description="Affected component or microservice")
    description: str = Field(description="Explanation of why this component failed")
    impact_percentage: float = Field(
        description="Estimated percentage contribution to total failure",
        ge=0.0,
        le=100.0,
    )

class IncidentReport(BaseModel):
    incident_id: str = Field(description="Unique tracking ID for the incident (e.g., INC-1042)")
    summary: str = Field(description="High-level overview of the incident", min_length=10)
    severity: SeverityLevel = Field(description="Incident severity level classification")
    downtime_minutes: int = Field(description="Total system downtime in minutes", ge=0)
    root_causes: List[RootCause] = Field(description="List of identified root causes")
    action_items: List[str] = Field(description="Immediate remediation steps and preventive actions")

class IncidentIngestRequest(BaseModel):
    raw_logs: str = Field(..., min_length=20, description="Raw incident logs or postmortem text")

# ─── 3. PARSING & DEFENSIVE CLEANING HELPERS ────────────────────────────────
def robust_json_parser(raw_response: str, model_cls: type[BaseModel]) -> BaseModel:
    """Strips markdown fences and parses string into a validated Pydantic model."""
    cleaned = raw_response.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        parsed_dict = json.loads(cleaned)
        return model_cls.model_validate(parsed_dict)
    except (json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"Schema validation failed for {model_cls.__name__}: {error}")

async def parse_incident(log_text: str, client: AsyncOpenAI) -> IncidentReport:
    """Invokes LLM with strict native structured output format."""
    completion = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a Site Reliability Engineering lead. Extract incident postmortems into strict structured data.",
            },
            {
                "role": "user",
                "content": log_text,
            },
        ],
        response_format=IncidentReport,
        temperature=0.0,
    )
    return completion.choices[0].message.parsed

# ─── 4. API ENDPOINTS ────────────────────────────────────────────────────────
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "HEALTHY", "service": "Structured Extraction Gateway"}

@app.post(
    "/incidents/extract",
    response_model=IncidentReport,
    status_code=status.HTTP_200_OK,
)
async def extract_incident(
    payload: IncidentIngestRequest,
    client: AsyncOpenAI = Depends(get_openai_client),
):
    try:
        extract = await parse_incident(payload.raw_logs, client)
        return extract
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream model extraction error: {str(err)}",
        )
