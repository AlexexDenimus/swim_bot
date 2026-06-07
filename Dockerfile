FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Системные библиотеки для OpenCV и ffmpeg.
# libxcb1 и x11/gl нужны, чтобы не падать с ошибкой libxcb.so.1.
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

# rtmlib тянет OpenCV с GUI. Удаляем его и оставляем headless-сборку без X-сервера.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install --force-reinstall --no-deps \
    opencv-python-headless==4.13.0.92 \
    opencv-contrib-python-headless==4.13.0.92 \
    && pip install --no-cache-dir -e .

COPY . .

CMD ["python", "src/scripts/main.py"]
