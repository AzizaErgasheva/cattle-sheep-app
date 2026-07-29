from fastapi import APIRouter

from app.api.schemas import ModelInfoResponse
from app.config import get_settings, load_metadata

router = APIRouter()


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    settings = get_settings()
    meta = load_metadata()
    return ModelInfoResponse(
        model_name=meta.get("best_model_name", "unknown"),
        classes=list(settings.class_names),
        img_size=list(settings.img_size),
        test_accuracy=meta.get("test_accuracy"),
        test_f1=meta.get("test_f1"),
    )
