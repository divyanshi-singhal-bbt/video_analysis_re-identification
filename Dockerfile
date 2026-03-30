# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Footfall Detection — Streamlit App                                         ║
# ║  Base: python:3.10-slim  (~120 MB)                                          ║
# ║  torch CPU-only          (~700 MB)                                          ║
# ║  Everything else         (~300 MB)                                          ║
# ║  Expected final size:    ~1.2–1.4 GB                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

FROM python:3.10-slim

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .

# Install numpy first (pinned <2.0) so torch sees the correct ABI
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy>=1.26.4,<2.0.0" && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2+cpu \
        torchvision==0.17.2+cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download YOLOv8n weights (~6 MB) ──────────────────────────────────────
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ── Copy application ───────────────────────────────────────────────────────────
COPY . .

# ── Streamlit config ───────────────────────────────────────────────────────────
RUN mkdir -p /root/.streamlit
RUN printf '\
[server]\n\
headless = true\n\
address = "0.0.0.0"\n\
port = 8501\n\
fileWatcherType = "none"\n\
maxUploadSize = 2048\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# ── Run ────────────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]