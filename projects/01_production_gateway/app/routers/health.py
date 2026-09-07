from fastapi import APIRouter, status
from app.config import get_settings
from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    settings = get_settings()
    return HealthResponse(
        status="HEALTHY",
        environment=settings.environment,
        service=settings.app_name,
    )