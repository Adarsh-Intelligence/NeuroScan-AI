"""Inference engine for NeuroScan AI.

Loads the trained EfficientNetB0 classification model and provides fast,
safe prediction and Grad-CAM visual explanation on MRI scans.
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config as C
from .utils import get_logger, load_json

log = get_logger("inference")


class NeuroScanEngine:
    """Thread-safe inference and Grad-CAM explanation engine."""

    def __init__(self, model_name: str = "efficientnet", target_layer: str = "efficientnetb0"):
        self.model_name = model_name
        self.target_layer = target_layer
        self.model_dir = C.MODELS_DIR / model_name
        self.model_path = self.model_dir / "model.keras"

        if not self.model_path.exists():
            # Check for stageB_best or stageA_best fallback
            for alt in ["stageB_best.keras", "stageA_best.keras"]:
                if (self.model_dir / alt).exists():
                    self.model_path = self.model_dir / alt
                    break

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found in {self.model_dir}")

        # Load class mappings
        mapping_file = self.model_dir / "class_mapping.json"
        if mapping_file.exists():
            mapping = load_json(mapping_file)
            self.class_names = mapping.get("class_names", ["glioma", "meningioma", "no_tumor", "pituitary"])
        else:
            self.class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

        self.class_labels = {
            "glioma": "Glioma",
            "meningioma": "Meningioma",
            "no_tumor": "No Tumor (Healthy)",
            "pituitary": "Pituitary Tumor",
        }

        # Lazy model storage
        self._model = None
        self._warmup_done = False

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        import keras

        log.info(f"Loading Keras model from {self.model_path}...")
        self._model = keras.models.load_model(str(self.model_path))
        log.info(f"Model '{self._model.name}' successfully loaded.")

    def warmup(self):
        """Run a dummy forward pass to eliminate cold-start latency."""
        if not self._warmup_done:
            import gc
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = self.model(dummy, training=False)
            self._warmup_done = True
            gc.collect()
            log.info("Model warmup completed (memory freed).")

    def preprocess_bytes(self, image_bytes: bytes, target_size=(224, 224)) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess raw image bytes -> (float32 array [0,255] shape (224,224,3), original uint8 RGB array)."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                rgb_img = pil_img.convert("RGB")
                orig_arr = np.asarray(rgb_img, dtype=np.uint8)
                resized = rgb_img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
                arr_224 = np.asarray(resized, dtype=np.float32)
                return arr_224, orig_arr
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Uploaded file is not a valid or readable image: {e}") from e

    def generate_gradcam(
        self, img_224: np.ndarray, class_index: Optional[int] = None
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Generate Grad-CAM heatmap for the given 224x224 RGB image."""
        import tensorflow as tf

        x = tf.convert_to_tensor(img_224[None].astype("float32"))
        with tf.GradientTape() as tape:
            h, feats = x, None
            for layer in self.model.layers[1:]:  # skip InputLayer
                h = layer(h, training=False)
                if layer.name == self.target_layer:
                    feats = h
                    tape.watch(feats)
            probs = h
            if feats is None:
                # Fallback: if layer name doesn't match directly, try to search for the functional backbone
                for layer in self.model.layers:
                    if hasattr(layer, "layers"):
                        feats = layer.output
                        tape.watch(feats)
                        break
            if feats is None:
                raise ValueError(f"Target feature layer '{self.target_layer}' not found")
            ci = int(tf.argmax(probs[0])) if class_index is None else class_index
            score = probs[:, ci]

        grads = tape.gradient(score, feats)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1)).numpy()
        cam = cam / (cam.max() + 1e-8)
        return cam, ci, probs[0].numpy()

    def overlay_heatmap(self, orig_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
        """Create RGB overlay combining original image and Grad-CAM color map."""
        h, w = orig_rgb.shape[:2]
        heat = cv2.resize(cam.astype("float32"), (w, h), interpolation=cv2.INTER_LINEAR)
        heat_color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[..., ::-1]  # BGR to RGB
        blended = np.clip((1 - alpha) * orig_rgb + alpha * heat_color, 0, 255).astype(np.uint8)
        return blended

    def image_to_base64(self, img_rgb: np.ndarray, format: str = "JPEG") -> str:
        """Convert an RGB numpy array to base64 data URI string."""
        pil_img = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format=format, quality=88)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "jpeg" if format.upper() == "JPEG" else format.lower()
        return f"data:image/{mime};base64,{b64_str}"

    def validate_brain_mri(self, orig_rgb: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate whether the image adheres to radiometric and anatomical Brain MRI criteria.

        Checks:
        1. Chromatic saturation: clinical MRI acquisitions are grayscale.
        2. Scanner background: corners must exhibit the dark air region typical of scanner bores.
        3. Dynamic range: soft brain tissue contrast check.
        4. Cranial contour & centering: detects convex skull mask with appropriate area and position.
        """
        if orig_rgb.ndim != 3 or orig_rgb.shape[2] != 3:
            return False, "Invalid image tensor dimensions.", {}

        # 1. Color saturation (MRIs are grayscale acquisitions)
        r = orig_rgb[:, :, 0].astype(float)
        g = orig_rgb[:, :, 1].astype(float)
        b = orig_rgb[:, :, 2].astype(float)
        color_diff = float(np.mean(np.abs(r - g) + np.abs(r - b) + np.abs(g - b)))
        if color_diff > 12.0:
            return (
                False,
                "Color photograph detected. Authentic Brain MRI scans are grayscale radiological acquisitions.",
                {"color_diff": round(color_diff, 2)},
            )

        gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # 2. Border/Corner background check (dark scanner air background around head)
        ch, cw = max(int(h * 0.06), 3), max(int(w * 0.06), 3)
        corners = np.concatenate([
            gray[:ch, :cw].flatten(),
            gray[:ch, -cw:].flatten(),
            gray[-ch:, :cw].flatten(),
            gray[-ch:, -cw:].flatten(),
        ])
        corner_mean = float(np.mean(corners))
        if corner_mean > 85.0:
            return (
                False,
                "Missing dark scanner background. Everyday photos, documents, and scenes lack the dark air boundary of cranial MRI scans.",
                {"corner_mean": round(corner_mean, 2)},
            )

        center = gray[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]
        center_mean = float(np.mean(center))
        img_std = float(np.std(gray))

        if center_mean < 15.0:
            return False, "Image is underexposed or mostly blank dark pixels.", {"center_mean": round(center_mean, 2)}
        if center_mean > 240.0:
            return False, "Image is overexposed or blank white.", {"center_mean": round(center_mean, 2)}
        if img_std < 18.0:
            return False, "Image lacks the radiologic contrast of brain tissue.", {"contrast_std": round(img_std, 2)}

        # 3. Cranial anatomical contour & centering
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "No cranial anatomical contour found.", {}

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < (h * w * 0.06):
            return False, "Anatomical subject area is too small to be a cranial MRI slice.", {"area_ratio": round(area / (h * w), 3)}

        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        if solidity < 0.70:
            return (
                False,
                "Subject contour is fragmented or non-convex, unlike the natural elliptical skull boundary.",
                {"solidity": round(solidity, 3)},
            )

        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist_from_center = float(np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2))
            max_dim = max(h, w)
            if dist_from_center > 0.35 * max_dim:
                return False, "Cranial subject is not centered within the scanner frame.", {"dist_norm": round(dist_from_center / max_dim, 3)}

        return True, "Valid Brain MRI", {"color_diff": round(color_diff, 2), "corner_mean": round(corner_mean, 2), "solidity": round(solidity, 3)}

    def predict(
        self, image_bytes: bytes, include_gradcam: bool = True, enforce_mri_check: bool = True
    ) -> Dict[str, Any]:
        """Perform end-to-end inference on raw image bytes with MRI quality control."""
        t0 = time.perf_counter()
        img_224, orig_rgb = self.preprocess_bytes(image_bytes)

        # Brain MRI Quality Control / Out-of-Distribution Verification
        if enforce_mri_check:
            is_mri, rejection_reason, qc_stats = self.validate_brain_mri(orig_rgb)
            if not is_mri:
                orig_b64 = self.image_to_base64(orig_rgb)
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return {
                    "is_valid_mri": False,
                    "error_message": rejection_reason,
                    "predicted_class": None,
                    "predicted_label": "Not a Brain MRI",
                    "confidence": 0.0,
                    "probs": {},
                    "latency_ms": latency_ms,
                    "gradcam_base64": None,
                    "original_base64": orig_b64,
                    "qc_stats": qc_stats,
                    "disclaimer": "The uploaded image was rejected by the Brain MRI Quality Control filter."
                }

        # Direct forward pass without tf.data iterator allocation overhead
        batch = np.expand_dims(img_224, axis=0)
        raw_probs = self.model(batch, training=False).numpy()[0]

        top_idx = int(np.argmax(raw_probs))
        top_class = self.class_names[top_idx]
        top_conf = float(raw_probs[top_idx])

        # Probabilities dictionary formatted as {class_name: float}
        probs_dict = {
            cls: float(raw_probs[i]) for i, cls in enumerate(self.class_names)
        }

        # Generate Grad-CAM if requested
        gradcam_b64 = None
        if include_gradcam:
            try:
                cam, _, _ = self.generate_gradcam(img_224, class_index=top_idx)
                blended = self.overlay_heatmap(orig_rgb, cam)
                gradcam_b64 = self.image_to_base64(blended)
            except Exception as e:
                log.warning(f"Grad-CAM generation failed: {e}")

        orig_b64 = self.image_to_base64(orig_rgb)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        import gc
        gc.collect()

        return {
            "is_valid_mri": True,
            "error_message": None,
            "predicted_class": top_class,
            "predicted_label": self.class_labels.get(top_class, top_class.title()),
            "confidence": round(top_conf, 4),
            "probs": probs_dict,
            "latency_ms": latency_ms,
            "gradcam_base64": gradcam_b64,
            "original_base64": orig_b64,
            "disclaimer": "NeuroScan AI is a research decision-support tool, not an autonomous medical diagnostic device."
        }

