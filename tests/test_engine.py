import cv2
import numpy as np
from prometheus_client import REGISTRY

from gesture_captions.captions import CaptionBuilder
from gesture_captions.config import Settings
from gesture_captions.engine import CaptionEngine, annotate, read_video_frames
from gesture_captions.predictor import Prediction


class FixedPredictor:
    model_type = "test"
    words_added = 0

    def predict(self, frame_bgr):
        return Prediction("hello", 0.95)

    def on_word_added(self):
        self.words_added += 1


class BrokenPredictor(FixedPredictor):
    def predict(self, frame_bgr):
        raise RuntimeError("boom")


def metric(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_engine_updates_caption_and_metrics():
    before = metric("gesture_predictions_total", label="hello")
    predictor = FixedPredictor()
    engine = CaptionEngine(predictor, CaptionBuilder(stable_frames=2))
    frame = np.zeros((8, 8, 3), np.uint8)
    engine.process(frame)
    engine.process(frame)
    assert engine.captions.text == "hello"
    assert metric("gesture_predictions_total", label="hello") == before + 2
    assert predictor.words_added == 1


def test_engine_survives_inference_error():
    before = metric("gesture_inference_errors_total")
    engine = CaptionEngine(BrokenPredictor(), CaptionBuilder())
    assert engine.process(np.zeros((8, 8, 3), np.uint8)) is None
    assert metric("gesture_inference_errors_total") == before + 1


def test_annotate_does_not_change_input_frame():
    frame = np.zeros((120, 200, 3), np.uint8)
    output = annotate(frame, Prediction("hello", 0.9), "hello world")
    assert frame.sum() == 0
    assert output.sum() > 0
    assert annotate(frame, None, "").shape == frame.shape


def test_read_video_frames_samples_to_target_rate(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (32, 32))
    for _ in range(20):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()
    assert len(list(read_video_frames(path, target_fps=10))) == 10


def test_rtc_configuration_includes_turn_only_when_set():
    assert len(Settings().rtc_configuration()["iceServers"]) == 1
    settings = Settings(turn_url="turn:example.org", turn_username="u", turn_credential="p")
    servers = settings.rtc_configuration()["iceServers"]
    assert servers[1] == {"urls": ["turn:example.org"], "username": "u", "credential": "p"}


def test_describe_hides_turn_secrets():
    settings = Settings(turn_url="turn:example.org", turn_username="user1", turn_credential="pw123")
    assert "pw123" not in settings.describe()
    assert "user1" not in settings.describe()
