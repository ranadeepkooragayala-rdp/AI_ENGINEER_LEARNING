import json
import re
from typing import Dict, List
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.schemas import ServerMetricReport


def robust_json_parser(raw_response: str, model_cls: type[BaseModel]) -> BaseModel:
    """Strips Markdown fences and validates against Pydantic schema."""
    cleaned = raw_response.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        data = json.loads(cleaned)
        return model_cls.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as err:
        raise ValueError(f"Failed parsing {model_cls.__name__}: {err}")


async def extract_metrics_with_self_correction(
    raw_log: str,
    client: AsyncOpenAI,
    max_retries: int = 2,
) -> ServerMetricReport:
    """Runs autonomous reflection loop feeding validation errors back to the model."""
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an SRE telemetry parser. Extract metrics into a valid JSON object matching: "
                "server_name (str), cpu_usage (0.0-100.0), memory_usage (0.0-100.0), and status ('HEALTHY'|'WARNING'|'CRITICAL'). "
                "Output ONLY valid JSON."
            ),
        },
        {"role": "user", "content": raw_log},
    ]

    for attempt in range(max_retries + 1):
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
        )
        content = completion.choices[0].message.content or ""

        try:
            return robust_json_parser(content, ServerMetricReport)
        except ValueError as err:
            if attempt == max_retries:
                raise ValueError(f"Self-correction exhausted after {max_retries} attempts: {err}")

            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Output validation failed: {str(err)}. Return ONLY corrected JSON.",
            })