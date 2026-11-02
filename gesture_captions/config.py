"""Runtime settings, read from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    model_dir: str = "models"
    model_version: str = "dev"
    metrics_port: int = 9100
    stun_url: str = "stun:stun.l.google.com:19302"
    turn_url: str = ""
    turn_username: str = ""
    turn_credential: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_dir=os.getenv("MODEL_DIR", cls.model_dir),
            model_version=os.getenv("MODEL_VERSION", cls.model_version),
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
                "model_dir": self.model_dir,
                "model_version": self.model_version,
                "turn_configured": bool(self.turn_url),
            }
        )
