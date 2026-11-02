"""Streamlit UI for real-time hand gesture captions."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from gesture_captions import metrics
from gesture_captions.captions import CaptionBuilder
from gesture_captions.config import Settings
from gesture_captions.engine import CaptionEngine, annotate, read_video_frames
from gesture_captions.predictor import Predictor, load_predictor

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Gesture Captions", page_icon="🖐️", layout="centered")


@st.cache_resource
def get_settings() -> Settings:
    settings = Settings.from_env()
    logging.getLogger(__name__).info("Settings: %s", settings.describe())
    return settings


@st.cache_resource
def get_predictor(_settings: Settings) -> Predictor:
    predictor = load_predictor(_settings)
    metrics.set_model_info(predictor.model_type, _settings.model_version)
    return predictor


def new_engine(settings: Settings, predictor: Predictor) -> CaptionEngine:
    return CaptionEngine(
        predictor, CaptionBuilder(settings.confidence_threshold, settings.stable_frames)
    )


settings = get_settings()
metrics.start_metrics_server(settings.metrics_port)
predictor = get_predictor(settings)

if "engine" not in st.session_state:
    st.session_state.engine = new_engine(settings, predictor)
engine: CaptionEngine = st.session_state.engine

st.title("🖐️ Gesture Captions")
st.caption("Translates hand gestures into text captions.")

if predictor.model_type == "placeholder":
    st.warning(
        "Placeholder model: this does not recognise gestures. Labels only reflect frame "
        "brightness. Add a trained model to `models/` to enable real predictions.",
        icon="⚠️",
    )

live_tab, video_tab, image_tab = st.tabs(["Live webcam", "Upload video", "Upload image"])

with live_tab:

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        prediction = engine.process(image)
        output = annotate(image, prediction, engine.captions.text)
        return av.VideoFrame.from_ndarray(output, format="bgr24")

    webrtc_streamer(
        key="gesture-captions",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        rtc_configuration=settings.rtc_configuration(),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    @st.fragment(run_every=1)
    def live_caption() -> None:
        st.markdown("**Caption**")
        st.code(
            engine.captions.text or "(waiting for a steady gesture)", language=None, wrap_lines=True
        )

    live_caption()
    if st.button("Clear caption"):
        engine.captions.clear()
        st.rerun()
    st.caption("The camera needs HTTPS or localhost. If it won't connect, use the upload tabs.")

with video_tab:
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "webm"])
    if uploaded_video is not None and st.button("Generate captions"):
        video_engine = new_engine(settings, predictor)
        suffix = Path(uploaded_video.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary_file:
            temporary_file.write(uploaded_video.getbuffer())
            temporary_file.flush()
            frame_count = 0
            try:
                with st.spinner("Processing video..."):
                    for video_frame in read_video_frames(temporary_file.name):
                        video_engine.process(video_frame)
                        frame_count += 1
            except ValueError:
                st.error("This video could not be read. Try an MP4 file.")
            else:
                st.success(f"Processed {frame_count} frames.")
                st.text_area(
                    "Caption",
                    video_engine.captions.text or "(no steady gesture detected)",
                    height=100,
                    disabled=True,
                )

with image_tab:
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image_engine = new_engine(settings, predictor)
        decoded = cv2.imdecode(np.frombuffer(uploaded_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            st.error("This image could not be read.")
        else:
            image_prediction = image_engine.process(decoded)
            st.image(annotate(decoded, image_prediction, ""), channels="BGR")
            if image_prediction is None:
                st.error("The model could not process this image.")

with st.sidebar:
    st.subheader("Model")
    st.write(f"Type: `{predictor.model_type}`")
    st.write(f"Version: `{settings.model_version}`")
    st.write(f"Confidence threshold: `{settings.confidence_threshold}`")
