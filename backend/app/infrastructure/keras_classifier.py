"""Concrete ClassifierPort implementation backed by a trained .keras model.

TensorFlow is imported lazily, inside _load(), rather than at module level.
This means importing this file -- and therefore the whole app package -- never
requires TensorFlow to be installed unless a prediction is actually made. Unit
tests for the domain/application layers stay fast and TF-free; only the real
integration path pays the TensorFlow import cost.
"""
from pathlib import Path

from app.domain.entities import ImageInput, Prediction
from app.domain.exceptions import ModelNotLoadedError
from app.domain.ports import ClassifierPort
from app.infrastructure.image_utils import decode_and_preprocess


class KerasClassifierAdapter(ClassifierPort):
    def __init__(self, model_path: str | Path, class_names: list[str], img_size: tuple[int, int]):
        if len(class_names) != 2:
            raise ValueError("KerasClassifierAdapter currently supports binary classifiers only")
        self._model_path = Path(model_path)
        # class_names must be ordered [negative_label, positive_label] to match
        # how the sigmoid output was trained (see notebook LABEL_MAP / label_id).
        self._class_names = class_names
        self._img_size = img_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if not self._model_path.exists():
            raise ModelNotLoadedError(f"Model file not found at {self._model_path}")
        import tensorflow as tf  # local import -- see module docstring
        self._model = tf.keras.models.load_model(self._model_path)
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_path.stem

    def predict(self, image: ImageInput) -> Prediction:
        model = self._load()
        batch = decode_and_preprocess(image, self._img_size)
        prob_positive = float(model.predict(batch, verbose=0)[0][0])

        negative_label, positive_label = self._class_names
        probabilities = {
            negative_label: 1.0 - prob_positive,
            positive_label: prob_positive,
        }
        label = max(probabilities, key=probabilities.get)
        return Prediction(label=label, confidence=probabilities[label], probabilities=probabilities)
