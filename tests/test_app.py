"""Run the Streamlit script headlessly to catch startup errors."""

from pathlib import Path

import streamlit as st
import streamlit_webrtc
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def run_app(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setenv("METRICS_PORT", "19100")
    # The WebRTC component needs a real Streamlit server, which AppTest doesn't provide.
    monkeypatch.setattr(streamlit_webrtc, "webrtc_streamer", lambda **_: None)
    return AppTest.from_file(APP, default_timeout=30).run()


def test_app_uses_real_models(monkeypatch):
    app = run_app(monkeypatch)
    assert not app.exception
    assert not app.warning


def test_app_warns_about_placeholder_without_models(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    app = run_app(monkeypatch)
    assert not app.exception
    assert "Placeholder model" in app.warning[0].value
