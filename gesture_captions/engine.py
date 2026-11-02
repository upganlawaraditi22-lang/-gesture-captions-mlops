"""Run a predictor on frames while updating captions and metrics."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from gesture_captions import metrics
from gesture_captions.captions import CaptionBuilder
from gesture_captions.predictor import Prediction, Predictor

logger = logging.getLogger(__name__)


class CaptionEngine:
    def __init__(self, predictor: Predictor, captions: CaptionBuilder):
        self.predictor = predictor
        self.captions = captions

    def process(self, frame_bgr: np.ndarray) -> Prediction | None:
        """Handle one frame. Returns None on failure so a live stream can keep running."""
        metrics.FRAMES.inc()
        start = time.perf_counter()
        try:
            prediction = self.predictor.predict(frame_bgr)
        except Exception:
            metrics.ERRORS.inc()
            logger.exception("Inference failed")
            return None
        metrics.LATENCY.observe(time.perf_counter() - start)
        metrics.PREDICTIONS.labels(label=prediction.label).inc()
        metrics.CONFIDENCE.observe(prediction.confidence)
        if self.captions.update(prediction):
            metrics.WORDS.inc()
            self.predictor.on_word_added()
        return prediction


def annotate(frame_bgr: np.ndarray, prediction: Prediction | None, caption: str) -> np.ndarray:
    """Draw the current prediction and the end of the caption on a copy of the frame."""
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    status = f"{prediction.label} {prediction.confidence:.0%}" if prediction else "inference error"
    cv2.rectangle(output, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(output, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if caption:
        max_characters = max(10, width // 14)
        visible = caption[-max_characters:]
        cv2.rectangle(output, (0, height - 44), (width, height), (0, 0, 0), -1)
        cv2.putText(
            output, visible, (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
        )
    return output


def read_video_frames(path: str | Path, target_fps: float = 30.0) -> Iterator[np.ndarray]:
    """Yield frames from a video file at about ``target_fps``."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {path}")
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or target_fps
        step = max(1, round(source_fps / target_fps))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            if index % step == 0:
                yield frame
            index += 1
    finally:
        capture.release()
