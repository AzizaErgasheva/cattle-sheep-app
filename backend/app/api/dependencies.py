"""Dependency providers for FastAPI's `Depends`.

Adapters are constructed lazily (imported inside the function body, cached
with lru_cache) rather than instantiated at module import time. This keeps
`import app.api.main` cheap and TensorFlow-free until a request actually
needs a model, and lets tests override these providers with fakes via
`app.dependency_overrides[...]` without ever touching TensorFlow or sqlite.
"""
from functools import lru_cache

from app.application.explain_use_case import ExplainPredictionUseCase
from app.application.history_use_cases import (
    ClearHistoryUseCase,
    ListHistoryUseCase,
    RecordPredictionUseCase,
)
from app.application.list_models_use_case import ListModelsUseCase
from app.application.predict_use_case import PredictImageUseCase
from app.config import get_settings
from app.domain.ports import HistoryRepositoryPort, ModelRegistryPort, ThumbnailerPort


@lru_cache
def get_model_registry() -> ModelRegistryPort:
    from app.infrastructure.model_registry import ModelRegistry

    settings = get_settings()
    return ModelRegistry(
        models_dir=settings.models_dir,
        metadata_path=settings.metadata_path,
        class_names=list(settings.class_names),
        img_size=settings.img_size,
    )


@lru_cache
def get_history_repository() -> HistoryRepositoryPort:
    from app.infrastructure.history_repository import SqliteHistoryRepository

    settings = get_settings()
    return SqliteHistoryRepository(db_path=settings.history_db_path)


@lru_cache
def get_thumbnailer() -> ThumbnailerPort:
    from app.infrastructure.image_utils import PillowThumbnailer

    return PillowThumbnailer()


def get_predict_use_case() -> PredictImageUseCase:
    return PredictImageUseCase(registry=get_model_registry())


def get_explain_use_case() -> ExplainPredictionUseCase:
    return ExplainPredictionUseCase(registry=get_model_registry())


def get_list_models_use_case() -> ListModelsUseCase:
    return ListModelsUseCase(registry=get_model_registry())


def get_record_history_use_case() -> RecordPredictionUseCase:
    return RecordPredictionUseCase(repository=get_history_repository(), thumbnailer=get_thumbnailer())


def get_list_history_use_case() -> ListHistoryUseCase:
    return ListHistoryUseCase(repository=get_history_repository())


def get_clear_history_use_case() -> ClearHistoryUseCase:
    return ClearHistoryUseCase(repository=get_history_repository())
