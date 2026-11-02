"""Model loading and inference: MediaPipe hand landmarks followed by an ONNX classifier."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from gesture_captions.config import Settings
from gesture_captions.features import extract_features, extract_frame_features, prep_sequence

logger = logging.getLogger(__name__)

NO_HAND_LABEL = "NOTHING"
BUFFERING_LABEL = "BUFFERING"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class Mode:
    title: str
    model_name: str
    threshold: float
    stable_frames: int


# Thresholds match the training project's live demo. Motion uses 9 frames because that demo needs
# 3 agreeing predictions made every third frame.
MODES = {
    "static": Mode("Static signs", "static_mlp", 0.75, 12),
    "spell": Mode("Finger spelling", "spell_mlp", 0.75, 12),
    "motion": Mode("Motion signs", "seq_lstm", 0.80, 9),
}


class Predictor(Protocol):
    model_type: str
    threshold: float
    stable_frames: int

    def predict(self, frame_bgr: np.ndarray) -> Prediction: ...

    def display_frame(self, frame_bgr: np.ndarray) -> np.ndarray: ...

    def on_word_added(self) -> None: ...


class PlaceholderPredictor:
    """Stand-in used when model files are missing. It does NOT recognise gestures."""

    model_type = "placeholder"
    threshold = 0.7
    stable_frames = 5

    def predict(self, frame_bgr: np.ndarray) -> Prediction:
        brightness = float(np.mean(frame_bgr))
        if brightness < 85:
            return Prediction("DEMO_DARK", 0.9)
        if brightness < 170:
            return Prediction("DEMO_MEDIUM", 0.9)
        return Prediction("DEMO_BRIGHT", 0.9)

    def display_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr

    def on_word_added(self) -> None:
        pass


class OnnxClassifier:
    """A trained classifier. It's safe to share one instance between sessions."""

    def __init__(self, model_name: str, model_dir: str | Path):
        import onnxruntime as ort

        directory = Path(model_dir)
        with np.load(directory / f"{model_name}.npz", allow_pickle=False) as data:
            self.labels = tuple(str(label) for label in data["labels"])
            self.mean = data["mean"].astype(np.float32)
            self.std = data["std"].astype(np.float32)
            self.frames = int(data["frames"])
        self._session = ort.InferenceSession(
            str(directory / f"{model_name}.onnx"), providers=["CPUExecutionProvider"]
        )

    def classify(self, features: np.ndarray) -> Prediction:
        batch = ((features - self.mean) / self.std).astype(np.float32)[None]
        logits = np.asarray(self._session.run(None, {"x": batch})[0][0], dtype=np.float64)
        exponentials = np.exp(logits - logits.max())
        probabilities = exponentials / exponentials.sum()
        index = int(np.argmax(probabilities))
        return Prediction(self.labels[index], float(probabilities[index]))


@cache
def _load_classifier(model_name: str, model_dir: str) -> OnnxClassifier:
    return OnnxClassifier(model_name, model_dir)


def _create_hands(static_image: bool):
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=static_image,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )


class SignPredictor:
    """Tracks hands in one stream, so each session or upload needs its own instance."""

    model_type = "mediapipe-onnx"

    def __init__(self, classifier: OnnxClassifier, mode: Mode, hands):
        self._classifier = classifier
        self._hands = hands
        self._is_motion = classifier.frames > 0
        self._buffer: deque[np.ndarray] = deque(maxlen=max(classifier.frames, 1))
        self._results = None
        self.threshold = mode.threshold
        self.stable_frames = mode.stable_frames

    def predict(self, frame_bgr: np.ndarray) -> Prediction:
        # The models were trained on mirrored frames, and mirroring swaps left and right hands.
        rgb = cv2.cvtColor(cv2.flip(frame_bgr, 1), cv2.COLOR_BGR2RGB)
        self._results = self._hands.process(rgb)
        if self._is_motion:
            self._buffer.append(extract_frame_features(self._results))
            if len(self._buffer) < self._classifier.frames:
                return Prediction(BUFFERING_LABEL, 0.0)
            return self._classifier.classify(prep_sequence(np.stack(self._buffer)))
        features = extract_features(self._results)
        if features is None:
            return Prediction(NO_HAND_LABEL, 1.0)
        return self._classifier.classify(features)

    def display_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        import mediapipe as mp

        output = cv2.flip(frame_bgr, 1)
        if self._results is not None and self._results.multi_hand_landmarks:
            for landmarks in self._results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    output, landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
        return output

    def on_word_added(self) -> None:
        self._buffer.clear()


def models_available(settings: Settings) -> bool:
    directory = Path(settings.model_dir)
    return all(
        (directory / f"{mode.model_name}{suffix}").exists()
        for mode in MODES.values()
        for suffix in (".onnx", ".npz")
    )


def load_predictor(settings: Settings, mode: str = "static", static_image: bool = False):
    """Load a predictor, or the placeholder if the model files are missing.

    Invalid model files raise an error, so a bad image fails its rollout.
    """
    if not models_available(settings):
        logger.warning("No models in %s; using PlaceholderPredictor", settings.model_dir)
        return PlaceholderPredictor()
    selected = MODES[mode]
    classifier = _load_classifier(selected.model_name, str(Path(settings.model_dir)))
    return SignPredictor(classifier, selected, _create_hands(static_image))
