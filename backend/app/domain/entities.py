"""Core domain objects. Plain dataclasses -- no framework or ML library imports here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageInput:
    """Raw image bytes plus the filename they were uploaded with."""
    content: bytes
    filename: str


@dataclass(frozen=True)
class Prediction:
    """Result of classifying a single image with a specific model."""
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str


@dataclass(frozen=True)
class Explanation:
    """A prediction plus a Grad-CAM heatmap overlay rendered as PNG bytes."""
    prediction: Prediction
    overlay_png: bytes


@dataclass(frozen=True)
class ModelSummary:
    """Metadata about one available model, for the model selector / dashboard."""
    name: str
    display_name: str
    is_best: bool
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


@dataclass(frozen=True)
class HistoryEntry:
    """One past prediction, kept for the history panel."""
    id: str
    created_at: str  # ISO-8601 UTC timestamp
    model_name: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    thumbnail_data_url: str

