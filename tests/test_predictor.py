from types import SimpleNamespace

import numpy as np
import pytest

from gesture_captions.config import Settings
from gesture_captions.features import (
    FEATURE_SIZE,
    FRAME_FEATURE_SIZE,
    extract_features,
    extract_frame_features,
    prep_sequence,
)
from gesture_captions.predictor import (
    BUFFERING_LABEL,
    MODES,
    NO_HAND_LABEL,
    OnnxClassifier,
    PlaceholderPredictor,
    SignPredictor,
    load_predictor,
    models_available,
)


def fake_results(label="Right", offset=0.0):
    rng = np.random.default_rng(1)
    points = rng.uniform(0.2, 0.6, size=(21, 3)) + offset
    landmarks = SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y, z=z) for x, y, z in points])
    handedness = SimpleNamespace(classification=[SimpleNamespace(label=label)])
    return SimpleNamespace(multi_hand_landmarks=[landmarks], multi_handedness=[handedness])


class FakeHands:
    def __init__(self, results):
        self.results = results

    def process(self, frame):
        return self.results


NO_HAND = SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
FRAME = np.zeros((48, 64, 3), np.uint8)


def test_placeholder_is_labelled_demo():
    assert PlaceholderPredictor().predict(FRAME).label == "DEMO_DARK"


def test_features_use_the_right_hand_slot():
    right = extract_features(fake_results("Right"))
    left = extract_features(fake_results("Left"))
    assert right.shape == (FEATURE_SIZE,)
    assert right[:63].any() and not right[63:].any()
    assert left[63:].any() and not left[:63].any()
    assert extract_features(NO_HAND) is None


def test_prep_sequence_centres_wrist_motion():
    sequence = np.stack([extract_frame_features(fake_results(offset=i * 0.01)) for i in range(10)])
    assert sequence.shape == (10, FRAME_FEATURE_SIZE)
    prepared = prep_sequence(sequence)
    assert abs(prepared[:, FEATURE_SIZE : FEATURE_SIZE + 2].mean()) < 1e-5


@pytest.mark.parametrize("mode", list(MODES))
def test_committed_model_files_load_and_classify(mode):
    classifier = OnnxClassifier(MODES[mode].model_name, "models")
    assert NO_HAND_LABEL in classifier.labels
    shape = (classifier.frames, FRAME_FEATURE_SIZE) if classifier.frames else (FEATURE_SIZE,)
    prediction = classifier.classify(np.zeros(shape, np.float32))
    assert prediction.label in classifier.labels
    assert 0 < prediction.confidence <= 1


def test_static_predictor_returns_a_trained_label():
    classifier = OnnxClassifier("static_mlp", "models")
    predictor = SignPredictor(classifier, MODES["static"], FakeHands(fake_results()))
    assert predictor.predict(FRAME).label in classifier.labels


def test_static_predictor_reports_no_hand():
    classifier = OnnxClassifier("static_mlp", "models")
    predictor = SignPredictor(classifier, MODES["static"], FakeHands(NO_HAND))
    assert predictor.predict(FRAME).label == NO_HAND_LABEL


def test_motion_predictor_buffers_then_classifies_and_resets():
    classifier = OnnxClassifier("seq_lstm", "models")
    predictor = SignPredictor(classifier, MODES["motion"], FakeHands(fake_results()))
    for _ in range(classifier.frames - 1):
        assert predictor.predict(FRAME).label == BUFFERING_LABEL
    assert predictor.predict(FRAME).label in classifier.labels
    predictor.on_word_added()
    assert predictor.predict(FRAME).label == BUFFERING_LABEL


def test_predictor_mirrors_frames_like_training():
    seen = []

    class RecordingHands:
        def process(self, frame):
            seen.append(frame)
            return NO_HAND

    frame = np.zeros((4, 4, 3), np.uint8)
    frame[:, 0] = (255, 255, 255)
    classifier = OnnxClassifier("static_mlp", "models")
    SignPredictor(classifier, MODES["static"], RecordingHands()).predict(frame)
    assert seen[0][0, -1].sum() > 0
    assert seen[0][0, 0].sum() == 0


def test_real_mediapipe_runs_on_a_blank_frame():
    settings = Settings(model_dir="models")
    assert models_available(settings)
    assert load_predictor(settings, "static").predict(FRAME).label == NO_HAND_LABEL


def test_load_uses_placeholder_without_models(tmp_path):
    settings = Settings(model_dir=str(tmp_path))
    assert load_predictor(settings).model_type == "placeholder"
