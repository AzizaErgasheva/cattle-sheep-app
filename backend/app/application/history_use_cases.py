import uuid
from datetime import datetime, timezone

from app.domain.entities import HistoryEntry, ImageInput, Prediction
from app.domain.ports import HistoryRepositoryPort, ThumbnailerPort

THUMBNAIL_SIZE = (96, 96)


class RecordPredictionUseCase:
    """Saves a prediction (with a small thumbnail of the source image) to history."""

    def __init__(self, repository: HistoryRepositoryPort, thumbnailer: ThumbnailerPort):
        self._repository = repository
        self._thumbnailer = thumbnailer

    def execute(self, image: ImageInput, prediction: Prediction) -> HistoryEntry:
        entry = HistoryEntry(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            model_name=prediction.model_name,
            label=prediction.label,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            thumbnail_data_url=self._thumbnailer.to_data_url(image, THUMBNAIL_SIZE),
        )
        self._repository.add(entry)
        return entry


class ListHistoryUseCase:
    def __init__(self, repository: HistoryRepositoryPort):
        self._repository = repository

    def execute(self, limit: int = 20) -> list[HistoryEntry]:
        return self._repository.list_recent(limit)


class ClearHistoryUseCase:
    def __init__(self, repository: HistoryRepositoryPort):
        self._repository = repository

    def execute(self) -> None:
        self._repository.clear()
