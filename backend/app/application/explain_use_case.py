from app.domain.entities import Explanation, ImageInput
from app.domain.ports import ModelRegistryPort


class ExplainPredictionUseCase:
    """Produces a prediction *and* a Grad-CAM overlay for the same image and model."""

    def __init__(self, registry: ModelRegistryPort):
        self._registry = registry

    def execute(self, image: ImageInput, model_name: str | None = None) -> Explanation:
        resolved_name = model_name or self._registry.default_model_name()
        handle = self._registry.get(resolved_name)
        prediction = handle.predict(image)
        overlay_png = handle.explain(image)
        return Explanation(prediction=prediction, overlay_png=overlay_png)
