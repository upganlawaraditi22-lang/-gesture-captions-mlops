"""Runtime settings, read from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


def _float_in_range(name: str, default: float, low: float, high: float) -> float:
    value = float(os.getenv(name, default))
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    model_path: str = "models/gesture_model.onnx"
    labels_path: str = "models/labels.txt"
    model_version: str = "dev"
    confidence_threshold: float = 0.7
    stable_frames: int = 5
    metrics_port: int = 9100
    stun_url: str = "stun:stun.l.google.com:19302"
    turn_url: str = ""
    turn_username: str = ""
    turn_credential: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_path=os.getenv("MODEL_PATH", cls.model_path),
            labels_path=os.getenv("LABELS_PATH", cls.labels_path),
            model_version=os.getenv("MODEL_VERSION", cls.model_version),
            confidence_threshold=_float_in_range(
                "CONFIDENCE_THRESHOLD", cls.confidence_threshold, 0.0, 1.0
            ),
            stable_frames=_positive_int("STABLE_FRAMES", cls.stable_frames),
            metrics_port=_positive_int("METRICS_PORT", cls.metrics_port),
            stun_url=os.getenv("STUN_URL", cls.stun_url),
            turn_url=os.getenv("TURN_URL", ""),
            turn_username=os.getenv("TURN_USERNAME", ""),
            turn_credential=os.getenv("TURN_CREDENTIAL", ""),
        )

    def rtc_configuration(self) -> dict:
        """Build the browser's WebRTC ICE configuration."""
        servers: list[dict] = []
        if self.stun_url:
            servers.append({"urls": [self.stun_url]})
        if self.turn_url:
            servers.append(
                {
                    "urls": [self.turn_url],
                    "username": self.turn_username,
                    "credential": self.turn_credential,
                }
            )
        return {"iceServers": servers}

    def describe(self) -> str:
        """Return non-secret settings for logs."""
        return json.dumps(
            {
                "model_path": self.model_path,
                "model_version": self.model_version,
                "confidence_threshold": self.confidence_threshold,
                "stable_frames": self.stable_frames,
                "turn_configured": bool(self.turn_url),
            }
        )
