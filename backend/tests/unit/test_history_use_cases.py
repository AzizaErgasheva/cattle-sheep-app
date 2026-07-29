from app.application.history_use_cases import ClearHistoryUseCase, ListHistoryUseCase, RecordPredictionUseCase
from app.domain.entities import HistoryEntry, ImageInput, Prediction
from app.domain.ports import HistoryRepositoryPort, ThumbnailerPort


class FakeThumbnailer(ThumbnailerPort):
    def to_data_url(self, image: ImageInput, size: tuple[int, int]) -> str:
        return "data:image/jpeg;base64,ZmFrZQ=="


class FakeHistoryRepository(HistoryRepositoryPort):
    def __init__(self):
        self.entries: list[HistoryEntry] = []

    def add(self, entry: HistoryEntry) -> None:
        self.entries.append(entry)

    def list_recent(self, limit: int) -> list[HistoryEntry]:
        return self.entries[:limit]

    def clear(self) -> None:
        self.entries.clear()


def test_record_prediction_use_case_saves_entry_with_thumbnail():
    repo = FakeHistoryRepository()
    use_case = RecordPredictionUseCase(repository=repo, thumbnailer=FakeThumbnailer())
    prediction = Prediction(label="cow", confidence=0.9, probabilities={"cow": 0.9, "sheep": 0.1}, model_name="resnet50")

    entry = use_case.execute(ImageInput(content=b"x", filename="a.jpg"), prediction)

    assert len(repo.entries) == 1
    assert entry.label == "cow"
    assert entry.model_name == "resnet50"
    assert entry.thumbnail_data_url.startswith("data:image/jpeg;base64,")


def test_list_history_use_case_delegates_to_repository():
    repo = FakeHistoryRepository()
    repo.entries.append(
        HistoryEntry(
            id="1", created_at="now", model_name="resnet50", label="cow",
            confidence=0.9, probabilities={"cow": 0.9, "sheep": 0.1}, thumbnail_data_url="x",
        )
    )
    use_case = ListHistoryUseCase(repository=repo)

    result = use_case.execute(limit=10)

    assert len(result) == 1
    assert result[0].id == "1"


def test_clear_history_use_case_empties_repository():
    repo = FakeHistoryRepository()
    repo.entries.append(
        HistoryEntry(
            id="1", created_at="now", model_name="resnet50", label="cow",
            confidence=0.9, probabilities={"cow": 0.9}, thumbnail_data_url="x",
        )
    )
    use_case = ClearHistoryUseCase(repository=repo)

    use_case.execute()

    assert repo.entries == []
