from fastapi import APIRouter, Depends

from app.api.dependencies import get_model_registry
from app.api.schemas import HealthResponse
from app.domain.exceptions import ModelNotFoundError
from app.domain.ports import ModelRegistryPort

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(registry: ModelRegistryPort = Depends(get_model_registry)) -> HealthResponse:
    try:
        default_model = registry.default_model_name()
        return HealthResponse(status="ok", model=default_model)
    except ModelNotFoundError:
        return HealthResponse(status="degraded", model="none")
