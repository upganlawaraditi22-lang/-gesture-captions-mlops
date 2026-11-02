"""Streamlit UI that turns hand signs into captions and sentences."""

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
from gesture_captions.language import LANG_NAMES, LANGS, gloss_text, translate
from gesture_captions.predictor import MODES, load_predictor

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Gesture Captions", page_icon="🖐️", layout="centered")


@st.cache_resource
def get_settings() -> Settings:
    settings = Settings.from_env()
    logging.getLogger(__name__).info("Settings: %s", settings.describe())
    return settings


def new_engine(settings: Settings, mode: str, static_image: bool = False) -> CaptionEngine:
    predictor = load_predictor(settings, mode, static_image)
    metrics.set_model_info(predictor.model_type, settings.model_version)
    return CaptionEngine(predictor, CaptionBuilder(predictor.threshold, predictor.stable_frames))


def show_caption(words: list[str], language: str) -> None:
    if not words:
        st.code("(waiting for a steady sign)", language=None, wrap_lines=True)
        return
    st.markdown(f"**Signs:** {gloss_text(words)}")
    st.code(translate(words, language), language=None, wrap_lines=True)


settings = get_settings()
metrics.start_metrics_server(settings.metrics_port)

with st.sidebar:
    st.subheader("Settings")
    mode = st.selectbox("Sign type", list(MODES), format_func=lambda key: MODES[key].title)
    language = st.selectbox("Language", LANGS, format_func=lambda key: LANG_NAMES[key])

if st.session_state.get("engine_mode") != mode:
    st.session_state.engine = new_engine(settings, mode)
    st.session_state.engine_mode = mode
engine: CaptionEngine = st.session_state.engine

st.title("🖐️ Gesture Captions")
st.caption("Translates hand signs into English, Hindi, or Marathi captions.")

if engine.predictor.model_type == "placeholder":
    st.warning(
        "Placeholder model: this does not recognise gestures because the model files are missing.",
        icon="⚠️",
    )

live_tab, video_tab, image_tab = st.tabs(["Live webcam", "Upload video", "Upload image"])

with live_tab:

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        prediction = engine.process(image)
        output = annotate(
            engine.predictor.display_frame(image), prediction, gloss_text(engine.captions.words)
        )
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
        show_caption(engine.captions.words, language)

    live_caption()
    if st.button("Clear caption"):
        engine.captions.clear()
        st.rerun()
    st.caption("The camera needs HTTPS or localhost. If it won't connect, use the upload tabs.")

with video_tab:
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "webm"])
    if uploaded_video is not None and st.button("Generate captions"):
        video_engine = new_engine(settings, mode)
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
                show_caption(video_engine.captions.words, language)

with image_tab:
    if mode == "motion":
        st.info("Motion signs need video. Choose another sign type to use a single image.")
    else:
        uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image_bytes = np.frombuffer(uploaded_image.getvalue(), np.uint8)
            decoded = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if decoded is None:
                st.error("This image could not be read.")
            else:
                image_engine = new_engine(settings, mode, static_image=True)
                image_prediction = image_engine.process(decoded)
                st.image(
                    annotate(image_engine.predictor.display_frame(decoded), image_prediction, ""),
                    channels="BGR",
                )
                if image_prediction is None:
                    st.error("The model could not process this image.")
