from dataclasses import asdict

from fastapi import APIRouter, Depends

from app.api.dependencies import get_list_models_use_case, get_model_registry
from app.api.schemas import ModelsListResponse, ModelSummaryResponse
from app.application.list_models_use_case import ListModelsUseCase
from app.domain.ports import ModelRegistryPort

router = APIRouter()


@router.get("/models", response_model=ModelsListResponse)
async def list_models(
    use_case: ListModelsUseCase = Depends(get_list_models_use_case),
    registry: ModelRegistryPort = Depends(get_model_registry),
) -> ModelsListResponse:
    summaries = use_case.execute()
    default_model = registry.default_model_name() if summaries else ""
    return ModelsListResponse(
        models=[ModelSummaryResponse(**asdict(s)) for s in summaries],
        default_model=default_model,
    )
