"""Run the Streamlit script headlessly to catch startup errors."""

from pathlib import Path

import streamlit_webrtc
from streamlit.testing.v1 import AppTest


def test_app_starts_and_warns_about_placeholder(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.onnx"))
    monkeypatch.setenv("METRICS_PORT", "19100")
    # The WebRTC component needs a real Streamlit server, which AppTest doesn't provide.
    monkeypatch.setattr(streamlit_webrtc, "webrtc_streamer", lambda **_: None)
    app = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert "Placeholder model" in app.warning[0].value
