"""Interfaces the application layer depends on. Infrastructure implements these --
the domain never imports Keras, FastAPI, sqlite3, or anything else concrete."""
from abc import ABC, abstractmethod

from app.domain.entities import HistoryEntry, ImageInput, ModelSummary, Prediction


class ClassifierPort(ABC):
    """Anything that can turn an image into a Prediction."""

    @abstractmethod
    def predict(self, image: ImageInput) -> Prediction:
        ...


class ExplainerPort(ABC):
    """Anything that can render a visual explanation (e.g. Grad-CAM) for an image."""

    @abstractmethod
    def explain(self, image: ImageInput) -> bytes:
        """Returns a PNG-encoded overlay image."""
        ...


class ModelHandle(ClassifierPort, ExplainerPort):
    """A single loaded model that can both classify and explain itself.

    Combining both ports on one object means the underlying weights are
    loaded once and shared, instead of a classifier and an explainer each
    loading their own separate copy of the same model.
    """


class ModelRegistryPort(ABC):
    """Looks up a `ModelHandle` by name and lists what's available, for the
    model-selector UI and the analytics dashboard."""

    @abstractmethod
    def get(self, model_name: str) -> ModelHandle:
        ...

    @abstractmethod
    def available_models(self) -> list[ModelSummary]:
        ...

    @abstractmethod
    def default_model_name(self) -> str:
        """The best-performing model, used when the caller doesn't specify one."""
        ...


class ThumbnailerPort(ABC):
    """Renders a small preview of an uploaded image, for the history panel."""

    @abstractmethod
    def to_data_url(self, image: ImageInput, size: tuple[int, int]) -> str:
        ...


class HistoryRepositoryPort(ABC):
    """Persists and retrieves past predictions for the history panel."""

    @abstractmethod
    def add(self, entry: HistoryEntry) -> None:
        ...

    @abstractmethod
    def list_recent(self, limit: int) -> list[HistoryEntry]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...
