import json

import pytest

from app.domain.exceptions import ModelNotFoundError
from app.infrastructure.model_registry import ModelRegistry


def _make_registry(tmp_path, metadata: dict | None = None, model_files=("resnet50", "custom_cnn")):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name in model_files:
        (models_dir / f"{name}.keras").write_bytes(b"not a real model, just a placeholder")

    metadata_path = tmp_path / "metadata.json"
    if metadata is not None:
        metadata_path.write_text(json.dumps(metadata))

    return ModelRegistry(
        models_dir=models_dir,
        metadata_path=metadata_path,
        class_names=["cow", "sheep"],
        img_size=(224, 224),
    )


def test_available_models_merges_metadata_metrics(tmp_path):
    metadata = {
        "best_model_name": "resnet50",
        "models": [
            {"name": "resnet50", "display_name": "ResNet50", "accuracy": 0.977, "f1": 0.977},
            {"name": "custom_cnn", "display_name": "Custom CNN", "accuracy": 0.823, "f1": 0.822},
        ],
    }
    registry = _make_registry(tmp_path, metadata=metadata)

    summaries = registry.available_models()

    by_name = {s.name: s for s in summaries}
    assert by_name["resnet50"].is_best is True
    assert by_name["resnet50"].accuracy == 0.977
    assert by_name["custom_cnn"].is_best is False
    assert by_name["custom_cnn"].accuracy == 0.823


def test_available_models_without_metadata_still_lists_files(tmp_path):
    registry = _make_registry(tmp_path, metadata=None)

    summaries = registry.available_models()

    assert {s.name for s in summaries} == {"resnet50", "custom_cnn"}
    assert all(s.accuracy is None for s in summaries)
    assert all(s.is_best is False for s in summaries)


def test_default_model_name_uses_metadata_best_when_present(tmp_path):
    registry = _make_registry(tmp_path, metadata={"best_model_name": "resnet50"})

    assert registry.default_model_name() == "resnet50"


def test_default_model_name_falls_back_when_metadata_missing(tmp_path):
    registry = _make_registry(tmp_path, metadata=None, model_files=("custom_cnn",))

    assert registry.default_model_name() == "custom_cnn"


def test_get_raises_model_not_found_for_unknown_name(tmp_path):
    registry = _make_registry(tmp_path)

    with pytest.raises(ModelNotFoundError):
        registry.get("does_not_exist")


def test_default_model_name_raises_when_no_models_present(tmp_path):
    registry = _make_registry(tmp_path, metadata=None, model_files=())

    with pytest.raises(ModelNotFoundError):
        registry.default_model_name()
