from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_name: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model: str


class ModelSummaryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    display_name: str
    is_best: bool
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class ModelsListResponse(BaseModel):
    models: list[ModelSummaryResponse]
    default_model: str


class HistoryEntryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    created_at: str
    model_name: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    thumbnail_data_url: str


class HistoryListResponse(BaseModel):
    entries: list[HistoryEntryResponse]
