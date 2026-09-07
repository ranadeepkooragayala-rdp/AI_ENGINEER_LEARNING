from typing import Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["HEALTHY", "DEGRADED", "UNHEALTHY"]
    environment: str
    service: str


class ServerMetricReport(BaseModel):
    server_name: str = Field(..., description="Hostname or instance identifier")
    cpu_usage: float = Field(..., ge=0.0, le=100.0, description="CPU percentage utilization")
    memory_usage: float = Field(..., ge=0.0, le=100.0, description="Memory percentage utilization")
    status: Literal["HEALTHY", "WARNING", "CRITICAL"] = Field(..., description="Operational status")


class ExtractionRequest(BaseModel):
    raw_log: str = Field(..., min_length=15, description="Raw telemetry log text")


class StreamRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt text to generate tokens for")