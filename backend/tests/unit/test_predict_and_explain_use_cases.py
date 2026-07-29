from app.application.explain_use_case import ExplainPredictionUseCase
from app.application.predict_use_case import PredictImageUseCase
from app.domain.entities import Explanation, ImageInput, ModelSummary, Prediction
from app.domain.ports import ModelHandle, ModelRegistryPort

PNG_MAGIC_BYTES = b"\x89PNG\r\n\x1a\n"


class FakeModelHandle(ModelHandle):
    def __init__(self, name: str, label: str = "cow", confidence: float = 0.9):
        self._name = name
        self._label = label
        self._confidence = confidence

    def predict(self, image: ImageInput) -> Prediction:
        other = 1.0 - self._confidence
        other_label = "sheep" if self._label == "cow" else "cow"
        return Prediction(
            label=self._label,
            confidence=self._confidence,
            probabilities={self._label: self._confidence, other_label: other},
            model_name=self._name,
        )

    def explain(self, image: ImageInput) -> bytes:
        return PNG_MAGIC_BYTES


class FakeRegistry(ModelRegistryPort):
    def __init__(self):
        self._handles = {
            "resnet50": FakeModelHandle("resnet50", label="cow", confidence=0.98),
            "custom_cnn": FakeModelHandle("custom_cnn", label="sheep", confidence=0.7),
        }

    def get(self, model_name: str) -> ModelHandle:
        return self._handles[model_name]

    def available_models(self) -> list[ModelSummary]:
        return [
            ModelSummary(name="resnet50", display_name="ResNet50", is_best=True, accuracy=0.977),
            ModelSummary(name="custom_cnn", display_name="Custom Cnn", is_best=False, accuracy=0.823),
        ]

    def default_model_name(self) -> str:
        return "resnet50"


def test_predict_use_case_uses_default_model_when_none_specified():
    use_case = PredictImageUseCase(registry=FakeRegistry())

    result = use_case.execute(ImageInput(content=b"x", filename="a.jpg"), model_name=None)

    assert result.model_name == "resnet50"
    assert result.label == "cow"


def test_predict_use_case_honors_explicit_model_choice():
    use_case = PredictImageUseCase(registry=FakeRegistry())

    result = use_case.execute(ImageInput(content=b"x", filename="a.jpg"), model_name="custom_cnn")

    assert result.model_name == "custom_cnn"
    assert result.label == "sheep"


def test_explain_use_case_returns_prediction_and_overlay_for_chosen_model():
    use_case = ExplainPredictionUseCase(registry=FakeRegistry())

    result: Explanation = use_case.execute(ImageInput(content=b"x", filename="a.jpg"), model_name="custom_cnn")

    assert result.prediction.model_name == "custom_cnn"
    assert result.overlay_png == PNG_MAGIC_BYTES
