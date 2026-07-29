from app.application.explain_use_case import ExplainPredictionUseCase
from app.domain.entities import ImageInput, Prediction
from app.domain.ports import ClassifierPort, ExplainerPort

PNG_MAGIC_BYTES = b"\x89PNG\r\n\x1a\n"


class FakeClassifier(ClassifierPort):
    @property
    def model_name(self) -> str:
        return "fake"

    def predict(self, image: ImageInput) -> Prediction:
        return Prediction(label="sheep", confidence=0.8, probabilities={"cow": 0.2, "sheep": 0.8})


class FakeExplainer(ExplainerPort):
    def explain(self, image: ImageInput) -> bytes:
        return PNG_MAGIC_BYTES


def test_explain_use_case_returns_prediction_and_overlay():
    use_case = ExplainPredictionUseCase(classifier=FakeClassifier(), explainer=FakeExplainer())

    result = use_case.execute(ImageInput(content=b"fake-bytes", filename="sheep.jpg"))

    assert result.prediction.label == "sheep"
    assert result.overlay_png == PNG_MAGIC_BYTES
