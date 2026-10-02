# Web dashboard + REST API (/api/predict photo analysis) on CPU.
#
#   docker build -t face-mask-detection .
#   docker run -p 5000:5000 -e FMD_ACCESS_KEY=choose-a-long-key face-mask-detection
#   curl -H "X-Access-Key: choose-a-long-key" -F image=@photo.jpg http://127.0.0.1:5000/api/predict
#
# Without FMD_ACCESS_KEY anyone who can reach the port can use the API.
#
# The live webcam stream needs a camera device inside the container
# (Linux hosts: docker run --device /dev/video0 ...).
#
# Google Cloud Run: python deploy_cloud_run.py. Hugging Face Spaces: python deploy_hf_space.py (see README).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=5000

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Headless OpenCV: no GUI windows are opened by the web app
RUN pip install tensorflow opencv-python-headless numpy scipy scikit-learn \
        flask matplotlib seaborn tqdm Pillow gunicorn

COPY . .

RUN useradd --create-home appuser \
 && mkdir -p logs screenshots \
 && chown -R appuser logs screenshots
USER appuser

# Hosts that assign the port themselves (Railway, Render, ...) set $PORT
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/model' % os.environ.get('PORT', '5000'), timeout=4)"

# One worker so the model is loaded once; /api/predict serialises inference anyway
CMD ["sh", "-c", "exec gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} app:app"]
