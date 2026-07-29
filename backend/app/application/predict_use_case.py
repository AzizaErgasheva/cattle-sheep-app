"""Application layer -- orchestrates domain ports. Depends only on interfaces,
never on a concrete classifier, so it can be unit-tested with a fake registry."""
from app.domain.entities import ImageInput, Prediction
from app.domain.ports import ModelRegistryPort


class PredictImageUseCase:
    def __init__(self, registry: ModelRegistryPort):
        self._registry = registry

    def execute(self, image: ImageInput, model_name: str | None = None) -> Prediction:
        resolved_name = model_name or self._registry.default_model_name()
        handle = self._registry.get(resolved_name)
        return handle.predict(image)
