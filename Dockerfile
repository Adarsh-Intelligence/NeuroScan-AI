# ==============================================================================
# NeuroScan AI - Production Dockerfile
# Optimized for Brain Tumor MRI Deep Learning Inference & Grad-CAM Analysis
# ==============================================================================
FROM python:3.10-slim-bullseye

# Prevent python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

# Install required system packages for OpenCV and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install CPU-optimized TensorFlow for container deployment
RUN pip install --no-cache-dir "tensorflow-cpu==2.18.*"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and UI
COPY src/ /app/src/
COPY artifacts/models/efficientnet/ /app/artifacts/models/efficientnet/
COPY artifacts/metrics/efficientnet/ /app/artifacts/metrics/efficientnet/
COPY index.html /app/index.html
COPY app.py /app/app.py
COPY run_server.py /app/run_server.py

# Create a non-privileged user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose HTTP port
EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
