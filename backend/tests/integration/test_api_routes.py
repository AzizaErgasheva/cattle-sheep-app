from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_clear_history_use_case,
    get_explain_use_case,
    get_list_history_use_case,
    get_list_models_use_case,
    get_model_registry,
    get_predict_use_case,
    get_record_history_use_case,
)
from app.api.main import app
from app.application.explain_use_case import ExplainPredictionUseCase
from app.application.history_use_cases import ClearHistoryUseCase, ListHistoryUseCase, RecordPredictionUseCase
from app.application.list_models_use_case import ListModelsUseCase
from app.application.predict_use_case import PredictImageUseCase
from app.domain.entities import HistoryEntry, ImageInput, ModelSummary, Prediction
from app.domain.ports import HistoryRepositoryPort, ModelHandle, ModelRegistryPort, ThumbnailerPort

PNG_MAGIC_BYTES = b"\x89PNG\r\n\x1a\n"


class FakeModelHandle(ModelHandle):
    def predict(self, image: ImageInput) -> Prediction:
        return Prediction(label="sheep", confidence=0.87, probabilities={"cow": 0.13, "sheep": 0.87}, model_name="resnet50")

    def explain(self, image: ImageInput) -> bytes:
        return PNG_MAGIC_BYTES


class FakeRegistry(ModelRegistryPort):
    def get(self, model_name: str) -> ModelHandle:
        return FakeModelHandle()

    def available_models(self) -> list[ModelSummary]:
        return [
            ModelSummary(name="resnet50", display_name="ResNet50", is_best=True, accuracy=0.977, f1=0.977),
            ModelSummary(name="custom_cnn", display_name="Custom Cnn", is_best=False, accuracy=0.823, f1=0.822),
        ]

    def default_model_name(self) -> str:
        return "resnet50"


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


fake_registry = FakeRegistry()
fake_history_repo = FakeHistoryRepository()

app.dependency_overrides[get_model_registry] = lambda: fake_registry
app.dependency_overrides[get_predict_use_case] = lambda: PredictImageUseCase(registry=fake_registry)
app.dependency_overrides[get_explain_use_case] = lambda: ExplainPredictionUseCase(registry=fake_registry)
app.dependency_overrides[get_list_models_use_case] = lambda: ListModelsUseCase(registry=fake_registry)
app.dependency_overrides[get_record_history_use_case] = lambda: RecordPredictionUseCase(
    repository=fake_history_repo, thumbnailer=FakeThumbnailer()
)
app.dependency_overrides[get_list_history_use_case] = lambda: ListHistoryUseCase(repository=fake_history_repo)
app.dependency_overrides[get_clear_history_use_case] = lambda: ClearHistoryUseCase(repository=fake_history_repo)

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "resnet50"}


def test_models_endpoint_lists_all_with_best_flagged():
    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["default_model"] == "resnet50"
    names = {m["name"] for m in body["models"]}
    assert names == {"resnet50", "custom_cnn"}
    best = next(m for m in body["models"] if m["name"] == "resnet50")
    assert best["is_best"] is True
    assert best["accuracy"] == 0.977


def test_predict_endpoint_accepts_model_name_and_records_history():
    fake_history_repo.clear()
    files = {"file": ("cow.jpg", b"fake-image-bytes", "image/jpeg")}

    response = client.post("/predict", files=files, data={"model_name": "resnet50"})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "sheep"
    assert body["model_name"] == "resnet50"
    assert len(fake_history_repo.entries) == 1
    assert fake_history_repo.entries[0].label == "sheep"


def test_predict_endpoint_works_without_model_name():
    files = {"file": ("cow.jpg", b"fake-image-bytes", "image/jpeg")}

    response = client.post("/predict", files=files)

    assert response.status_code == 200


def test_explain_endpoint_returns_png_with_headers():
    files = {"file": ("cow.jpg", b"fake-image-bytes", "image/jpeg")}

    response = client.post("/predict/explain", files=files, data={"model_name": "resnet50"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-predicted-label"] == "sheep"
    assert response.content == PNG_MAGIC_BYTES


def test_history_endpoint_lists_and_clears():
    fake_history_repo.clear()
    files = {"file": ("cow.jpg", b"fake-image-bytes", "image/jpeg")}
    client.post("/predict", files=files)

    list_response = client.get("/history")
    assert list_response.status_code == 200
    assert len(list_response.json()["entries"]) == 1

    delete_response = client.delete("/history")
    assert delete_response.status_code == 204

    list_after_clear = client.get("/history")
    assert list_after_clear.json()["entries"] == []
