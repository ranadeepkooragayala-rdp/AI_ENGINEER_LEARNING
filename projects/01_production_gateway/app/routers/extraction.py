from fastapi import APIRouter, Depends, HTTPException, status
from openai import AsyncOpenAI

from app.dependencies import get_llm_client, rate_limiter
from app.schemas import ExtractionRequest, ServerMetricReport
from app.services.extractor import extract_metrics_with_self_correction

router = APIRouter(prefix="/v1", tags=["Extraction"])


@router.post(
    "/extract",
    response_model=ServerMetricReport,
    status_code=status.HTTP_200_OK,
)
async def extract_endpoint(
    payload: ExtractionRequest,
    client: AsyncOpenAI = Depends(get_llm_client),
    _rate_limit: bool = Depends(rate_limiter(max_requests=5, window_seconds=60)),
):
    try:
        return await extract_metrics_with_self_correction(payload.raw_log, client)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(err),
        )