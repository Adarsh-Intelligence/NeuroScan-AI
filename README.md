# 🧠 NeuroScan-AI: Brain Tumor MRI Classification

[![Developer](https://img.shields.io/badge/Developer-Adarsh%20Dwivedi-blue?style=for-the-badge&logo=github)](https://github.com/Adarsh-Intelligence)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](tests/)

A leakage-safe deep learning pipeline that classifies brain MRI scans into four categories, with transfer learning, honest evaluation, and Grad-CAM explainability.

| Class | Description |
|---|---|
| **Glioma** | Tumor arising from glial cells |
| **Meningioma** | Tumor of the meninges |
| **Pituitary** | Tumor of the pituitary gland |
| **No Tumor** | Healthy scan |

> ⚠️ **Disclaimer:** This is an educational / research project. It is **not** a medical device and must not be used for clinical diagnosis.

---

## ✨ Highlights

- **Leakage-safe splitting.** Roboflow exports contain augmented clones and near-duplicates. This pipeline detects them (MD5 + 64-bit DCT perceptual hash + source-stem matching), clusters them with Union-Find, and splits by *group* so no scan ever crosses a split boundary.
- **Two models, one protocol.** A custom 4-block CNN trained from scratch vs. a 2-stage fine-tuned EfficientNetB0, both selected on **validation macro-F1** and evaluated **once** on the held-out test set.
- **Explainability.** Grad-CAM heatmaps to sanity-check where the model is looking.
- **Reproducible.** Central config, fixed seed (`42`), saved splits, metrics, and plots under `artifacts/`, plus a pytest suite.

---

## 📊 Results

Test split: **384 images** (grouped, never seen during training or model selection). Model: **EfficientNetB0** (Stage A, frozen-backbone feature extraction selected on validation).

| Metric | Score |
|---|---|
| Accuracy | **89.84%** |
| Macro F1 | **0.894** |
| Weighted F1 | **0.899** |
| Macro Precision / Recall | 0.892 / 0.898 |

**Per-class performance**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Glioma | 0.976 | 0.910 | 0.942 | 134 |
| Meningioma | 0.789 | 0.807 | 0.798 | 88 |
| No Tumor | 0.924 | 0.913 | 0.918 | 80 |
| Pituitary | 0.878 | 0.963 | 0.919 | 82 |

**Efficiency (CPU)**

| Metric | Value |
|---|---|
| Parameters | 4.05 M |
| Model size | 16.3 MB |
| Single-image latency (median) | ~31 ms |
| Batched latency | ~13.8 ms / image |
| Training time | ~95 s |

<p align="center">
  <img src="artifacts/plots/efficientnet/test_confusion_matrix.png" alt="EfficientNetB0 test confusion matrix" width="480">
</p>

**Takeaway:** Meningioma is the hardest class. Most errors are glioma ↔ meningioma and meningioma → pituitary confusions, consistent with their visual overlap on MRI.

> Numbers above come from `artifacts/metrics/efficientnet/test_metrics.json`. Run `python -m src.compare_models` to regenerate the Custom CNN vs. EfficientNet comparison report for your own training run.

---

## 🗂️ Dataset

- **Total images:** 2,443 (0 unreadable, 0 label mismatches, 0 exact duplicates)
- **Class balance:** Glioma 805 · Pituitary 610 · Meningioma 545 · No Tumor 483 (max/min ratio 1.67, handled with class weights)
- **Hidden duplication found:** 2,091 unique source stems, 704 near-duplicate pairs, and **272 groups leaking across the original train/val/test split**. This is exactly why the pipeline re-splits by group.

| Split | Images | Share |
|---|---|---|
| Train | 1,699 | 69.6% |
| Validation | 360 | 14.7% |
| Test | 384 | 15.7% |

The raw dataset is **not committed**. Place it in a folder named `DATASET FOR NEROSCAN AI/` at the project root (see `src/config.py`). The folder is treated as read-only.

---

## 🏗️ Pipeline

```
 raw dataset folder ──► 1. discover & inspect ──► 2. find duplicates ──► 3. leakage-safe split
   (read-only)            (src/data_loader.py)      (pHash + MD5 + stem)     (70 / 15 / 15, grouped)
                                                                                  │
 6. Grad-CAM  ◄── 5. evaluate on TEST ◄── 4. train on TRAIN, select on VAL ◄──────┘
 (src/gradcam.py) (src/evaluate.py)       (src/train.py: Custom CNN | EfficientNetB0)
```

1. **Discovery** (`src/data_loader.py`): finds images across Roboflow-exported splits and extracts original stems to detect augmented clones.
2. **Integrity & duplicate grouping** (`src/data_loader.py`): MD5 byte checks and perceptual hashing (stricter threshold when labels differ across classes), clustered via Union-Find.
3. **Stratified grouped split** (`src/prepare_data.py`): `StratifiedGroupKFold` with 20 folds → 14 train / 3 val / 3 test.
4. **Training** (`src/train.py`, `src/models/`):
   - **Custom CNN:** VGG-style, 4 blocks, BatchNorm + Dropout + GlobalAveragePooling.
   - **EfficientNetB0:** Stage A trains the head on a frozen backbone; Stage B unfreezes the top blocks (BatchNorm kept frozen).
   - Anatomy-preserving augmentation (±10° rotation, ±8% shift, ±10% zoom, horizontal flip, mild brightness/contrast) on training data only.
5. **Evaluation** (`src/evaluate.py`, `src/compare_models.py`): accuracy, macro/weighted F1, confusion matrix, model size, and inference latency.
6. **Explainability** (`src/gradcam.py`): Grad-CAM overlays.

---

## 🚀 Getting Started

### 1. Install

```bash
git clone https://github.com/Adarsh-Intelligence/NeuroScan-AI.git
cd NeuroScan-AI

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# TensorFlow is installed separately:
pip install "tensorflow[and-cuda]==2.21.*"   # WSL2 / Linux GPU
pip install "tensorflow==2.21.*"             # Windows / CPU-only
```

Optional GPU check: `python -m src.gpu_check`

### 2. Run tests

```bash
python -m pytest -q
```

### 3. Inspect data & build splits

```bash
python -m src.explore
python -m src.prepare_data
```

### 4. Train

```bash
# Fast CPU smoke test
python -m src.train --model custom_cnn --smoke

# Full training
python -m src.train --model custom_cnn
python -m src.train --model efficientnet
```

### 5. Evaluate & compare

```bash
python -m src.evaluate --model efficientnet --split test
python -m src.compare_models
```

### 6. Grad-CAM

```bash
python -m src.gradcam --model efficientnet --n 8
```

---

## 📖 Walkthrough Notebook

For an end-to-end tour with diagrams, dataset integrity checks, and Grad-CAM visuals, open
👉 [`notebooks/01_project_walkthrough.ipynb`](notebooks/01_project_walkthrough.ipynb) (pre-executed, outputs included).

---

## 📁 Repository Structure

```
.
├── notebooks/
│   └── 01_project_walkthrough.ipynb   # Interactive walkthrough with outputs
├── src/
│   ├── config.py                      # Central configuration & hyperparameters
│   ├── data_loader.py                 # Dataset parsing, hashing & grouping
│   ├── explore.py                     # Dataset inspection & integrity report
│   ├── prepare_data.py                # Leakage-safe stratified split
│   ├── preprocessing.py               # Resizing, tf.data pipeline, class weights
│   ├── augmentation.py                # Anatomy-preserving augmentations
│   ├── models/
│   │   ├── custom_cnn.py              # 4-block CNN
│   │   └── efficientnet.py            # EfficientNetB0 transfer learning
│   ├── train.py                       # Training loop, callbacks, stage selection
│   ├── inference.py                   # Production inference & Grad-CAM engine
│   ├── fast_train_efficientnet.py     # Quick EfficientNet training script
│   ├── evaluate.py                    # Test metrics, confusion matrix, latency
│   ├── compare_models.py              # Markdown comparison report
│   ├── gradcam.py                     # Grad-CAM heatmaps & overlays
│   ├── plots.py / utils.py            # Plotting & helpers
│   └── gpu_check.py                   # GPU availability check
├── app.py                             # FastAPI production application & routes
├── frontend.html                      # Interactive web interface with Grad-CAM
├── run_server.py                      # CLI server launcher
├── Dockerfile                         # Production container image
├── docker-compose.yml                 # 1-command container orchestration
├── DEPLOYMENT.md                      # Detailed deployment guide
├── tests/                             # pytest test suite (including test_api.py)
└── artifacts/                         # Splits, models, metrics, plots, exploration
```

---

## 🌐 Web Application & REST API

NeuroScan AI includes an interactive, browser-based user interface and a FastAPI backend with live Grad-CAM heatmap visualization.

### Start the Local Server
```bash
python run_server.py --port 8000
```
- **Web Interface**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### Run with Docker Compose
```bash
docker compose up --build
```

For full deployment instructions on Render, Railway, Hugging Face, or Cloud VMs, see [DEPLOYMENT.md](DEPLOYMENT.md).


---

## 👨‍💻 Author

**Adarsh Dwivedi**
- GitHub: [@Adarsh-Intelligence](https://github.com/Adarsh-Intelligence)
- Repository: [NeuroScan-AI](https://github.com/Adarsh-Intelligence/NeuroScan-AI)

*Built as an end-to-end medical deep learning project for reliable brain tumor classification and explainability.*