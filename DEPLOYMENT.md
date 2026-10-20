# NeuroScan AI — Deployment & Backend Guide

NeuroScan AI includes a production-grade FastAPI backend serving both high-performance REST inference endpoints and an interactive frontend with Grad-CAM heatmaps.

---

## 🚀 Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Ensure TensorFlow is installed for your environment: `pip install tensorflow` or `pip install tensorflow-cpu`)*

### 2. Start the Server
```bash
python run_server.py --port 8000
```
Or with Uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open in Browser
- **Interactive UI**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🐳 Docker Deployment

### 1. Build and Run with Docker Compose
```bash
docker compose up --build -d
```
The application will be live at `http://localhost:8000`.

### 2. Standard Docker CLI
```bash
# Build Docker image
docker build -t neuroscan-ai .

# Run container
docker run -p 8000:8000 --name neuroscan-ai neuroscan-ai
```

---

## 📡 API Endpoints Reference

### `POST /api/predict`
Upload an MRI image and receive classification probabilities along with a Grad-CAM heatmap.

- **Content-Type**: `multipart/form-data`
- **Body**: `file` (binary image data, JPG or PNG)
- **Query Parameter**: `gradcam=true` (optional, default `true`)

**Example Response**:
```json
{
  "predicted_class": "glioma",
  "predicted_label": "Glioma",
  "confidence": 0.9729,
  "probs": {
    "glioma": 0.9729,
    "meningioma": 0.0227,
    "no_tumor": 0.0004,
    "pituitary": 0.0040
  },
  "latency_ms": 42.5,
  "gradcam_base64": "data:image/jpeg;base64,...",
  "original_base64": "data:image/jpeg;base64,...",
  "disclaimer": "NeuroScan AI is a research decision-support tool, not an autonomous medical diagnostic device."
}
```

### `POST /api/gradcam`
Stream the Grad-CAM blended overlay directly as a `JPEG` binary image.

### `GET /health`
Returns readiness probe status:
```json
{
  "status": "healthy",
  "service": "NeuroScan-AI",
  "version": "1.0.0",
  "model_loaded": true,
  "classes": ["glioma", "meningioma", "no_tumor", "pituitary"]
}
```

### `GET /api/info`
Returns model architecture specs, held-out test metrics, and training configuration.

---

## ☁️ Cloud Deployment Options

### Render (1-Click)
1. Fork or push this repository to GitHub.
2. In Render, select **New Web Service** and connect your repo.
3. Render will automatically detect [`render.yaml`](render.yaml) or use:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1`

### Railway / Heroku
The included [`Procfile`](Procfile) specifies the web worker command:
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
```

### Hugging Face Spaces (Docker Space)
1. Create a new Space with the **Docker** SDK.
2. Push this repo. The included [`Dockerfile`](Dockerfile) will build and serve on port 7860 or 8000 automatically.

---

## 🧪 Testing

Run automated backend tests:
```bash
pytest tests/test_api.py -v
```
