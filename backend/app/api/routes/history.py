from dataclasses import asdict

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import Response

from app.api.dependencies import get_clear_history_use_case, get_list_history_use_case
from app.api.schemas import HistoryEntryResponse, HistoryListResponse
from app.application.history_use_cases import ClearHistoryUseCase, ListHistoryUseCase
from app.config import get_settings

router = APIRouter()


@router.get("/history", response_model=HistoryListResponse)
async def list_history(
    limit: int = Query(default=None, ge=1, le=200),
    use_case: ListHistoryUseCase = Depends(get_list_history_use_case),
) -> HistoryListResponse:
    resolved_limit = limit or get_settings().history_default_limit
    entries = use_case.execute(limit=resolved_limit)
    return HistoryListResponse(entries=[HistoryEntryResponse(**asdict(e)) for e in entries])


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_history(use_case: ClearHistoryUseCase = Depends(get_clear_history_use_case)) -> Response:
    use_case.execute()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
