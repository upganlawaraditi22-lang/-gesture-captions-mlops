FROM python:3.12-slim

ARG MODEL_VERSION=dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    MODEL_VERSION=${MODEL_VERSION}

WORKDIR /app

# OpenCV needs these system libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit ./.streamlit
COPY gesture_captions ./gesture_captions
COPY models ./models
COPY app.py .

RUN useradd --uid 10001 --create-home appuser
USER 10001

EXPOSE 8501 9100

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"]

CMD ["python", "-m", "gesture_captions.serve"]
