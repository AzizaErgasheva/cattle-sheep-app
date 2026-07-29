"""Discovers trained models on disk, merges in whatever metrics metadata.json
provides, and lazily builds a KerasModelAdapter per model name -- loaded once,
cached, shared between predict and explain calls for that model.
"""
import json
from pathlib import Path

from app.domain.entities import ModelSummary
from app.domain.exceptions import ModelNotFoundError
from app.domain.ports import ModelHandle, ModelRegistryPort

_BACKBONE_PREPROCESS = {"resnet50", "mobilenet", "efficientnet"}  # custom_cnn (or anything else) -> flat model, no preprocess


class ModelRegistry(ModelRegistryPort):
    def __init__(
        self,
        models_dir: str | Path,
        metadata_path: str | Path,
        class_names: list[str],
        img_size: tuple[int, int],
    ):
        self._models_dir = Path(models_dir)
        self._metadata_path = Path(metadata_path)
        self._class_names = class_names
        self._img_size = img_size
        self._adapters: dict[str, ModelHandle] = {}

    def _discover_files(self) -> dict[str, Path]:
        if not self._models_dir.exists():
            return {}
        return {p.stem: p for p in sorted(self._models_dir.glob("*.keras"))}

    def _load_metadata(self) -> dict:
        if self._metadata_path.exists():
            try:
                return json.loads(self._metadata_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def get(self, model_name: str) -> ModelHandle:
        if model_name not in self._adapters:
            files = self._discover_files()
            if model_name not in files:
                available = ", ".join(sorted(files)) or "(none found)"
                raise ModelNotFoundError(f"Unknown model '{model_name}'. Available: {available}")

            from app.infrastructure.keras_model_adapter import KerasModelAdapter

            preprocess_fn_name = model_name if model_name in _BACKBONE_PREPROCESS else None
            self._adapters[model_name] = KerasModelAdapter(
                model_name=model_name,
                model_path=files[model_name],
                class_names=self._class_names,
                img_size=self._img_size,
                preprocess_fn_name=preprocess_fn_name,
            )
        return self._adapters[model_name]

    def default_model_name(self) -> str:
        meta = self._load_metadata()
        best = meta.get("best_model_name")
        files = self._discover_files()
        if best and best in files:
            return best
        if files:
            return next(iter(sorted(files)))
        raise ModelNotFoundError(f"No .keras models found in {self._models_dir}")

    def available_models(self) -> list[ModelSummary]:
        meta = self._load_metadata()
        metrics_by_name = {m["name"]: m for m in meta.get("models", [])}
        best_name = meta.get("best_model_name")
        files = self._discover_files()

        summaries = []
        for name in sorted(files):
            m = metrics_by_name.get(name, {})
            summaries.append(
                ModelSummary(
                    name=name,
                    display_name=m.get("display_name", name.replace("_", " ").title()),
                    is_best=(name == best_name),
                    accuracy=m.get("accuracy"),
                    precision=m.get("precision"),
                    recall=m.get("recall"),
                    f1=m.get("f1"),
                )
            )
        return summaries
