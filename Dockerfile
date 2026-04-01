FROM python:3.10-slim

# Minimal system deps for OpenCV headless
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 1. CPU-only torch — must come first so ultralytics doesn't pull CUDA build
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Install all deps except opencv
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    boto3 \
    matplotlib \
    scikit-learn \
    ultralytics

# 3. Force headless opencv AFTER ultralytics — overwrites the full opencv it pulled in
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]