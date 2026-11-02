"""Turn noisy per-frame predictions into a stable running caption."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from gesture_captions.predictor import Prediction

DEFAULT_NO_GESTURE_LABELS = frozenset({"nothing", "none", "blank", "idle"})


class CaptionBuilder:
    """Add a word only after a gesture is held steadily.

    A held gesture adds one word. Repeating a word requires lowering the hand or showing a
    different gesture in between. Brief low-confidence or missed frames do not cause repeats.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        stable_frames: int = 5,
        no_gesture_labels: Iterable[str] = DEFAULT_NO_GESTURE_LABELS,
        max_words: int = 40,
    ):
        self._threshold = confidence_threshold
        self._stable_frames = stable_frames
        self._no_gesture = {label.lower() for label in no_gesture_labels}
        self._max_words = max_words
        self._lock = threading.Lock()
        self._words: list[str] = []
        self._candidate: str | None = None
        self._count = 0
        self._last_added: str | None = None
        self._no_gesture_count = 0

    def update(self, prediction: Prediction) -> bool:
        """Process one prediction. Returns True if a word was added."""
        with self._lock:
            if prediction.label.lower() in self._no_gesture:
                self._candidate, self._count = None, 0
                self._no_gesture_count += 1
                if self._no_gesture_count >= self._stable_frames:
                    self._last_added = None
                return False
            self._no_gesture_count = 0
            if prediction.confidence < self._threshold:
                self._candidate, self._count = None, 0
                return False
            if prediction.label == self._candidate:
                self._count += 1
            else:
                self._candidate, self._count = prediction.label, 1
            if self._count < self._stable_frames or prediction.label == self._last_added:
                return False
            self._words.append(prediction.label)
            self._words = self._words[-self._max_words :]
            self._last_added = prediction.label
            return True

    @property
    def words(self) -> list[str]:
        with self._lock:
            return list(self._words)

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def clear(self) -> None:
        with self._lock:
            self._words.clear()
            self._candidate, self._count, self._last_added = None, 0, None
            self._no_gesture_count = 0
