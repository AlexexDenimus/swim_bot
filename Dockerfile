FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libraries required by OpenCV/ffmpeg at runtime.
# libxcb1 + the x11/gl libs fix the "libxcb.so.1: cannot open shared object file" crash.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxcb1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src/ src/

# rtmlib pulls in GUI OpenCV (opencv-python / opencv-contrib-python).
# Remove those and keep only the headless builds so cv2 never needs an X server.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install --force-reinstall --no-deps \
    opencv-python-headless==4.13.0.92 \
    opencv-contrib-python-headless==4.13.0.92 \
    && pip install --no-cache-dir -e .

COPY . .

CMD ["python", "src/scripts/main.py"]
