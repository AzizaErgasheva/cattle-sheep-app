"""Central configuration, overridable via APP_* environment variables."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/


class Settings(BaseSettings):
    models_dir: Path = BASE_DIR / "models"
    metadata_path: Path = BASE_DIR / "models" / "metadata.json"
    history_db_path: Path = BASE_DIR / "data" / "history.db"
    history_default_limit: int = 20

    img_size: tuple[int, int] = (224, 224)
    # Ordered [negative_label, positive_label] to match the sigmoid output
    # from training (see the notebook's LABEL_MAP).
    class_names: tuple[str, str] = ("cow", "sheep")

    cors_allow_origins: list[str] = ["*"]  # tighten to the deployed frontend origin in production

    # protected_namespaces=() silences pydantic's warning about fields named
    # `model_*` (model_config) colliding with its own namespace.
    model_config = SettingsConfigDict(env_prefix="APP_", protected_namespaces=())


@lru_cache
def get_settings() -> Settings:
    return Settings()
