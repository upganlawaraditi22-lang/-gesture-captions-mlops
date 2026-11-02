"""Prometheus metrics, served on a separate port from the Streamlit UI."""

from __future__ import annotations

import logging
import threading

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

FRAMES = Counter("gesture_frames_processed_total", "Frames sent to the model")
PREDICTIONS = Counter("gesture_predictions_total", "Predictions by gesture", ["label"])
ERRORS = Counter("gesture_inference_errors_total", "Frames that failed inference")
WORDS = Counter("gesture_caption_words_total", "Words added to captions")
LATENCY = Histogram(
    "gesture_inference_latency_seconds",
    "Time taken to run the model on one frame",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
CONFIDENCE = Histogram(
    "gesture_prediction_confidence",
    "Confidence of the top prediction",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
MODEL_INFO = Gauge("gesture_model_info", "Loaded model", ["model_type", "version"])

_lock = threading.Lock()
_started = False


def start_metrics_server(port: int) -> bool:
    """Start the metrics endpoint once per process. Streamlit reruns must not repeat it."""
    global _started
    with _lock:
        if _started:
            return False
        try:
            start_http_server(port)
        except OSError:
            logger.exception("Could not start metrics server on port %s", port)
            return False
        _started = True
        return True


def set_model_info(model_type: str, version: str) -> None:
    MODEL_INFO.labels(model_type=model_type, version=version).set(1)
