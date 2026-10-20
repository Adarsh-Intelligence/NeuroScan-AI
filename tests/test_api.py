"""Integration tests for the NeuroScan AI FastAPI backend."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import app
from src import config as C, preprocessing as P


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_mri_bytes():
    """Create a realistic dummy MRI slice (224x224 RGB image with circular head phantom)."""
    h, w = 224, 224
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    # Elliptical skull boundary
    mask = ((yy - 112) / 90) ** 2 + ((xx - 112) / 80) ** 2 <= 1.0
    arr[mask] = 120
    # Brain tissue variation
    inner = ((yy - 112) / 75) ** 2 + ((xx - 112) / 65) ** 2 <= 1.0
    arr[inner] = 180

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_serve_frontend(client: TestClient):
    """Verify that root endpoint serves the frontend HTML interface."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "NEUROSCAN" in resp.text


def test_health_check(client: TestClient):
    """Verify health endpoint reports healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "glioma" in data["classes"]


def test_model_info(client: TestClient):
    """Verify info endpoint returns model architecture and metrics."""
    resp = client.get("/api/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "EfficientNetB0" in data["model_name"]
    assert len(data["classes"]) == 4
    assert "accuracy" in data["test_metrics"]
    assert data["test_metrics"]["accuracy"] > 0.80


def test_predict_endpoint_valid_image(client: TestClient, sample_mri_bytes: bytes):
    """Verify prediction endpoint with a valid image."""
    files = {"file": ("test_mri.jpg", sample_mri_bytes, "image/jpeg")}
    resp = client.post("/api/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["predicted_class"] in ["glioma", "meningioma", "no_tumor", "pituitary"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probs" in data
    assert len(data["probs"]) == 4

    # Check sum of probabilities ≈ 1.0
    total_prob = sum(data["probs"].values())
    assert pytest.approx(total_prob, rel=1e-2) == 1.0

    # Check Grad-CAM base64 is present
    assert data["gradcam_base64"] is not None
    assert data["gradcam_base64"].startswith("data:image/jpeg;base64,")


def test_predict_endpoint_invalid_file(client: TestClient):
    """Verify rejection of corrupted / invalid image payload."""
    files = {"file": ("bad.txt", b"not an image", "text/plain")}
    resp = client.post("/api/predict", files=files)
    assert resp.status_code in [400, 422]


def test_gradcam_stream_endpoint(client: TestClient, sample_mri_bytes: bytes):
    """Verify direct Grad-CAM image streaming endpoint."""
    files = {"file": ("test_mri.jpg", sample_mri_bytes, "image/jpeg")}
    resp = client.post("/api/gradcam", files=files)
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "image/jpeg"
    assert len(resp.content) > 100


def test_predict_endpoint_rejects_non_mri_photo(client: TestClient):
    """Verify that a non-MRI photo (e.g. colored or non-cranial image) is flagged and rejected."""
    # Create a colored test image (e.g. RGB photo simulation)
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :, 0] = 220  # Red
    arr[:, :, 1] = 60   # Green
    arr[:, :, 2] = 40   # Blue
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    color_bytes = buf.getvalue()

    files = {"file": ("photo.jpg", color_bytes, "image/jpeg")}
    resp = client.post("/api/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid_mri"] is False
    assert "Color photograph detected" in data["error_message"]
    assert data["predicted_class"] is None

