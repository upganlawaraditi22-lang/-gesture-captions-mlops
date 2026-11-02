"""Model loading and inference.

The app only depends on ``Predictor.predict(frame) -> Prediction``. To plug in a
model that is not a plain image classifier (for example MediaPipe landmarks plus
a small classifier), add a class with that method and return it from
``load_predictor``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from gesture_captions.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class Predictor(Protocol):
    model_type: str
    labels: tuple[str, ...]

    def predict(self, frame_bgr: np.ndarray) -> Prediction: ...


class PlaceholderPredictor:
    """Stand-in used until a trained model is added. It does NOT recognise gestures.

    The label only reflects frame brightness, so the pipeline, captions, and
    metrics can be tested without a trained model.
    """

    model_type = "placeholder"
    labels = ("DEMO_DARK", "DEMO_MEDIUM", "DEMO_BRIGHT")

    def predict(self, frame_bgr: np.ndarray) -> Prediction:
        brightness = float(np.mean(frame_bgr))
        if brightness < 85:
            return Prediction("DEMO_DARK", 0.9)
        if brightness < 170:
            return Prediction("DEMO_MEDIUM", 0.9)
        return Prediction("DEMO_BRIGHT", 0.9)


class OnnxPredictor:
    """Image classifier exported to ONNX.

    Default preprocessing is RGB scaled to 0-1. If the model was trained with
    other preprocessing, change ``_preprocess`` to match or accuracy will suffer.
    """

    model_type = "onnx"

    def __init__(self, model_path: str | Path, labels: tuple[str, ...]):
        import onnxruntime as ort

        self.labels = labels
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        shape = model_input.shape
        if len(shape) != 4:
            raise ValueError(f"Expected a 4D image input, got shape {shape}")
        self._channels_first = shape[1] == 3
        if not self._channels_first and shape[3] != 3:
            raise ValueError(f"Could not find a 3-channel axis in input shape {shape}")
        height, width = (shape[2], shape[3]) if self._channels_first else (shape[1], shape[2])
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(f"Model input needs a fixed height and width, got shape {shape}")
        self._size = (width, height)

        output_shape = self._session.get_outputs()[0].shape
        classes = output_shape[-1]
        if isinstance(classes, int) and classes != len(labels):
            raise ValueError(f"Model outputs {classes} classes but labels file has {len(labels)}")

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, self._size), cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
        return np.transpose(batch, (0, 3, 1, 2)) if self._channels_first else batch

    def predict(self, frame_bgr: np.ndarray) -> Prediction:
        output = self._session.run(None, {self._input_name: self._preprocess(frame_bgr)})[0]
        scores = np.asarray(output, dtype=np.float64).reshape(-1)
        is_probability = scores.min() >= 0 and abs(float(scores.sum()) - 1.0) < 1e-3
        if not is_probability:
            exponentials = np.exp(scores - scores.max())
            scores = exponentials / exponentials.sum()
        index = int(np.argmax(scores))
        return Prediction(self.labels[index], float(scores[index]))


def read_labels(path: str | Path) -> tuple[str, ...]:
    labels = tuple(
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def load_predictor(settings: Settings) -> Predictor:
    """Load the trained model, or the placeholder when no model file exists.

    A model file that exists but is invalid raises an error, so a bad image fails
    its rollout instead of quietly serving the placeholder.
    """
    if not Path(settings.model_path).exists():
        logger.warning("No model at %s; using PlaceholderPredictor", settings.model_path)
        return PlaceholderPredictor()
    return OnnxPredictor(settings.model_path, read_labels(settings.labels_path))
