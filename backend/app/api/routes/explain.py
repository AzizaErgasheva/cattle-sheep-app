from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.api.dependencies import get_explain_use_case
from app.application.explain_use_case import ExplainPredictionUseCase
from app.domain.entities import ImageInput
from app.domain.exceptions import DomainError

router = APIRouter()


@router.post("/predict/explain")
async def explain(
    file: UploadFile = File(...),
    model_name: str | None = Form(default=None),
    use_case: ExplainPredictionUseCase = Depends(get_explain_use_case),
) -> Response:
    content = await file.read()
    image = ImageInput(content=content, filename=file.filename or "upload")

    try:
        explanation = use_case.execute(image, model_name=model_name)
    except DomainError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    headers = {
        "X-Predicted-Label": explanation.prediction.label,
        "X-Confidence": f"{explanation.prediction.confidence:.4f}",
        "X-Model-Name": explanation.prediction.model_name,
    }
    return Response(content=explanation.overlay_png, media_type="image/png", headers=headers)
