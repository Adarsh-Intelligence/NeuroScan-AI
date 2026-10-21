"""NeuroScan AI - Production FastAPI Backend Application.

Serves the REST API and the interactive frontend interface for brain tumor
MRI classification with Grad-CAM heatmaps.
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from src import config as C
from src.inference import NeuroScanEngine
from src.utils import get_logger, load_json

log = get_logger("app")

# Global inference engine instance
engine: Optional[NeuroScanEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initialize and warmup model on server startup."""
    global engine
    log.info("Starting up NeuroScan AI server...")
    try:
        engine = NeuroScanEngine()
        engine.warmup()
        log.info("NeuroScan AI backend is ready to accept requests.")
    except Exception as e:
        log.error(f"Failed to initialize NeuroScanEngine: {e}")
    yield
    log.info("Shutting down NeuroScan AI server...")


app = FastAPI(
    title="NeuroScan AI API",
    description="Deep learning brain tumor MRI classification with Grad-CAM explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _get_frontend_path() -> Path:
    for candidate in ("index.html", "frontend.html"):
        p = C.PROJECT_ROOT / candidate
        if p.exists():
            return p
    return C.PROJECT_ROOT / "index.html"


FRONTEND_PATH = _get_frontend_path()


@app.get("/", summary="Serve Interactive UI", response_class=FileResponse)
async def serve_frontend():
    """Serve the complete single-page interactive UI."""
    path = _get_frontend_path()
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend interface (index.html / frontend.html) not found on server root",
        )
    return FileResponse(str(path), media_type="text/html")


@app.get("/frontend.html", response_class=FileResponse)
@app.get("/index.html", response_class=FileResponse)
async def serve_frontend_alias():
    return await serve_frontend()


@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Health status and model readiness probe."""
    is_loaded = engine is not None and engine._model is not None
    return {
        "status": "healthy" if is_loaded else "initializing",
        "service": "NeuroScan-AI",
        "version": "1.0.0",
        "model_loaded": is_loaded,
        "classes": engine.class_names if engine else [],
    }


@app.get("/api/info", summary="Model metadata and performance metrics")
async def get_model_info() -> Dict[str, Any]:
    """Return model architecture, training configuration, and test metrics."""
    metrics_path = C.METRICS_DIR / "efficientnet" / "test_metrics.json"
    summary_path = C.MODELS_DIR / "efficientnet" / "training_summary.json"

    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    summary = load_json(summary_path) if summary_path.exists() else {}

    return {
        "model_name": "EfficientNetB0 (Fine-tuned Transfer Learning)",
        "input_shape": [224, 224, 3],
        "classes": engine.class_names if engine else ["glioma", "meningioma", "no_tumor", "pituitary"],
        "class_labels": engine.class_labels if engine else {},
        "test_metrics": {
            "accuracy": metrics.get("accuracy", 0.8255),
            "macro_f1": metrics.get("macro_f1", 0.8228),
            "macro_precision": metrics.get("macro_precision", 0.824),
            "macro_recall": metrics.get("macro_recall", 0.8357),
            "n_test_samples": metrics.get("n_samples", 384),
            "per_class": metrics.get("per_class", {}),
        },
        "train_info": {
            "train_size": summary.get("train_size", 1699),
            "val_size": summary.get("val_size", 360),
            "selected_stage": summary.get("selected_stage", "stageB_fine_tuning"),
        },
    }


@app.post("/api/predict", summary="Classify brain MRI scan")
async def predict_mri(
    file: UploadFile = File(..., description="Brain MRI image (JPEG or PNG)"),
    gradcam: bool = Query(True, description="Whether to include Grad-CAM heatmap visualization"),
):
    """Analyze an uploaded MRI scan image and return classification probabilities and Grad-CAM."""
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still initializing. Please retry in a few moments.",
        )

    # Validate content type
    if file.content_type and not (
        file.content_type.startswith("image/") or file.content_type in ["application/octet-stream"]
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Please upload a JPG or PNG image.",
        )

    image_bytes = await file.read()
    if not image_bytes or len(image_bytes) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty or corrupted file uploaded.",
        )

    try:
        result = engine.predict(image_bytes, include_gradcam=gradcam)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        log.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing the scan.",
        )


@app.post("/api/gradcam", summary="Generate and return Grad-CAM overlay image directly")
async def get_gradcam_image(
    file: UploadFile = File(..., description="Brain MRI image (JPEG or PNG)"),
):
    """Directly return the Grad-CAM blended overlay as a JPEG image stream."""
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready.",
        )

    image_bytes = await file.read()
    try:
        img_224, orig_rgb = engine.preprocess_bytes(image_bytes)
        cam, ci, _ = engine.generate_gradcam(img_224)
        blended = engine.overlay_heatmap(orig_rgb, cam)

        from PIL import Image

        pil_blended = Image.fromarray(blended)
        buf = io.BytesIO()
        pil_blended.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        log.error(f"Grad-CAM stream error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not generate Grad-CAM for this image.",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
