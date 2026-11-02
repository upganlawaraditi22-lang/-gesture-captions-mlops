import pytest
from prometheus_client import REGISTRY

from gesture_captions import metrics, serve
from gesture_captions.config import Settings


def test_prepare_publishes_model_info(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "start_metrics_server", lambda port: True)
    serve.prepare(Settings(model_path=str(tmp_path / "missing.onnx"), model_version="v-test"))
    value = REGISTRY.get_sample_value(
        "gesture_model_info", {"model_type": "placeholder", "version": "v-test"}
    )
    assert value == 1


def test_prepare_stops_on_invalid_model(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(metrics, "start_metrics_server", started.append)
    model_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "labels.txt"
    model_path.write_bytes(b"not a model")
    labels_path.write_text("a\n", encoding="utf-8")
    settings = Settings(model_path=str(model_path), labels_path=str(labels_path))
    with pytest.raises(Exception):  # noqa: B017 - onnxruntime raises its own error types
        serve.prepare(settings)
    assert started == []
