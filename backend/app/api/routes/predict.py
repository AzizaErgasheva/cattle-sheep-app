from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.dependencies import get_predict_use_case, get_record_history_use_case
from app.api.schemas import PredictionResponse
from app.application.history_use_cases import RecordPredictionUseCase
from app.application.predict_use_case import PredictImageUseCase
from app.domain.entities import ImageInput
from app.domain.exceptions import DomainError

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str | None = Form(default=None),
    use_case: PredictImageUseCase = Depends(get_predict_use_case),
    record_use_case: RecordPredictionUseCase = Depends(get_record_history_use_case),
) -> PredictionResponse:
    content = await file.read()
    image = ImageInput(content=content, filename=file.filename or "upload")

    try:
        prediction = use_case.execute(image, model_name=model_name)
    except DomainError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    record_use_case.execute(image, prediction)

    return PredictionResponse(
        label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        model_name=prediction.model_name,
    )
