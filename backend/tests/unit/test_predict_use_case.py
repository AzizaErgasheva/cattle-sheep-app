from app.application.predict_use_case import PredictImageUseCase
from app.domain.entities import ImageInput, Prediction
from app.domain.ports import ClassifierPort


class FakeClassifier(ClassifierPort):
    def __init__(self, prediction: Prediction):
        self._prediction = prediction

    @property
    def model_name(self) -> str:
        return "fake"

    def predict(self, image: ImageInput) -> Prediction:
        return self._prediction


def test_predict_use_case_delegates_to_classifier():
    expected = Prediction(label="cow", confidence=0.91, probabilities={"cow": 0.91, "sheep": 0.09})
    use_case = PredictImageUseCase(classifier=FakeClassifier(expected))

    result = use_case.execute(ImageInput(content=b"fake-bytes", filename="cow.jpg"))

    assert result == expected
