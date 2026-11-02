"""Turn MediaPipe hand landmarks into model features.

These functions match the training code. Changing them without retraining lowers accuracy.

Static gestures: one frame, 126 numbers (2 hands x 21 landmarks x [x, y, z]), with each hand moved
to the wrist origin and scaled to unit size.
Motion gestures: 30 frames of 130 numbers (the same 126 plus the wrist x and y of each hand).
"""

from __future__ import annotations

import numpy as np

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATS_PER_HAND = LANDMARKS_PER_HAND * 3
FEATURE_SIZE = NUM_HANDS * FEATS_PER_HAND
FRAME_FEATURE_SIZE = FEATURE_SIZE + 2 * NUM_HANDS


def normalize_hand(hand_landmarks) -> np.ndarray:
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    points -= points[0]
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 1e-6:
        points /= scale
    return points.flatten()


def _hand_slots(results):
    """Yield (slot, landmarks), where slot 0 is the right hand and slot 1 is the left."""
    used = set()
    for landmarks, handedness in zip(
        results.multi_hand_landmarks, results.multi_handedness, strict=False
    ):
        slot = 0 if handedness.classification[0].label == "Right" else 1
        if slot in used:
            slot = 1 - slot
        if slot in used:
            continue
        used.add(slot)
        yield slot, landmarks


def extract_features(results) -> np.ndarray | None:
    """Return static features, or None if no hand is visible."""
    if not results.multi_hand_landmarks:
        return None
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for slot, landmarks in _hand_slots(results):
        features[slot * FEATS_PER_HAND : (slot + 1) * FEATS_PER_HAND] = normalize_hand(landmarks)
    return features


def extract_frame_features(results) -> np.ndarray:
    """Return per-frame motion features. Missing hands are zeros."""
    features = np.zeros(FRAME_FEATURE_SIZE, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return features
    for slot, landmarks in _hand_slots(results):
        features[slot * FEATS_PER_HAND : (slot + 1) * FEATS_PER_HAND] = normalize_hand(landmarks)
        wrist = landmarks.landmark[0]
        start = FEATURE_SIZE + 2 * slot
        features[start : start + 2] = (wrist.x, wrist.y)
    return features


def prep_sequence(sequence) -> np.ndarray:
    """Make wrist positions relative to their average, so position in the frame doesn't matter."""
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    for slot in range(NUM_HANDS):
        hand = sequence[:, slot * FEATS_PER_HAND : (slot + 1) * FEATS_PER_HAND]
        present = np.abs(hand).sum(1) > 0
        columns = slice(FEATURE_SIZE + 2 * slot, FEATURE_SIZE + 2 * slot + 2)
        if present.any():
            sequence[present, columns] -= sequence[present, columns].mean(0)
            sequence[present, columns] *= 5.0
        sequence[~present, columns] = 0.0
    return sequence
