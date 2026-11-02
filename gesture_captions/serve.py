"""Container entry point.

Streamlit only runs ``app.py`` when a browser connects. Starting here means the
metrics endpoint is available immediately, and an invalid model stops the pod at
startup, which fails the Kubernetes rollout.
"""

from __future__ import annotations

import logging
import sys

from streamlit.web import cli as streamlit_cli

from gesture_captions import metrics
from gesture_captions.config import Settings
from gesture_captions.predictor import load_predictor


def prepare(settings: Settings) -> None:
    predictor = load_predictor(settings)
    metrics.set_model_info(predictor.model_type, settings.model_version)
    metrics.start_metrics_server(settings.metrics_port)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    prepare(Settings.from_env())
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
