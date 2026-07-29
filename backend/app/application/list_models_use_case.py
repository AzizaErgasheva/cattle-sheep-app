from app.domain.entities import ModelSummary
from app.domain.ports import ModelRegistryPort


class ListModelsUseCase:
    def __init__(self, registry: ModelRegistryPort):
        self._registry = registry

    def execute(self) -> list[ModelSummary]:
        return self._registry.available_models()
