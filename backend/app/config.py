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

    # Plain comma-separated string, NOT a JSON list. pydantic-settings parses
    # list[str] fields from env vars as strict JSON, which is fragile through
    # web-based env var editors (Render, Railway, etc. can mangle brackets/
    # quotes) -- a comma-separated string has no such failure mode.
    # e.g. "*" or "https://foo.vercel.app,https://bar.com"
    cors_allow_origins: str = "*"

    # protected_namespaces=() silences pydantic's warning about fields named
    # `model_*` (model_config) colliding with its own namespace.
    model_config = SettingsConfigDict(env_prefix="APP_", protected_namespaces=())

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()