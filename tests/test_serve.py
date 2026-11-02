from prometheus_client import REGISTRY

from gesture_captions import metrics, serve
from gesture_captions.config import Settings


def test_prepare_publishes_model_info(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "start_metrics_server", lambda port: True)
    serve.prepare(Settings(model_dir=str(tmp_path), model_version="v-test"))
    value = REGISTRY.get_sample_value(
        "gesture_model_info", {"model_type": "placeholder", "version": "v-test"}
    )
    assert value == 1
