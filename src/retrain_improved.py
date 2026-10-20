"""Improved EfficientNetB0 retraining script with full augmentation pipeline.

Root causes fixed compared to the previous fast_train_efficientnet.py:
1. Features were cached WITHOUT augmentation - this killed generalization.
   FIX: Use full tf.data pipeline with augmentation during training.
2. Fine-tuning stage B was either skipped or used too high a LR (1e-5).
   FIX: Two-stage training via src.train (head + proper fine-tune at 5e-6).
3. Manifest paths were broken (cross-project path issue).
   FIX: resolve_filepath() handles both PROJECT_ROOT and DATASET_DIR.parent.
4. No fine-tuning below block6 - only last 2 blocks were unfrozen.
   FIX: Unfreeze from block5 for more representational capacity.

Usage:
    python -m src.retrain_improved [--smoke]
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from . import config as C
from . import data_loader as D
from .augmentation import build_augmenter
from .preprocessing import compute_class_weights, make_dataset, preprocessing_spec
from .utils import get_logger, save_json, set_global_seed

log = get_logger("retrain_improved")


def main(smoke=False, batch_size=16):
    import keras
    import tensorflow as tf

    set_global_seed(C.SEED)
    C.ensure_dirs()

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    device_str = "GPU" if gpus else "CPU"
    log.info("TF %s | Keras %s | Device: %s", tf.__version__, keras.__version__, device_str)

    cfg = C.get_config()
    if smoke:
        cfg.efficientnet.head_epochs = 1
        cfg.efficientnet.finetune_epochs = 1
        batch_size = 4
    cfg.data.batch_size = batch_size

    class_names, mapping = D.load_class_mapping()
    train_df = D.load_split("train")
    val_df = D.load_split("val")
    test_df = D.load_split("test")

    if smoke:
        train_df = train_df.groupby("class_name", group_keys=False).apply(lambda g: g.head(8))
        val_df = val_df.groupby("class_name", group_keys=False).apply(lambda g: g.head(4))

    log.info("Dataset: %d train | %d val | %d test | classes: %s", len(train_df), len(val_df), len(test_df), class_names)

    cw = compute_class_weights(train_df["label"].to_numpy(), len(class_names))
    log.info("Class weights: %s", {class_names[i]: round(w, 3) for i, w in cw.items()})

    augmenter = build_augmenter(cfg.augment, seed=cfg.data.seed)
    train_ds = make_dataset(train_df, cfg.data.image_size, batch_size, training=True, augmenter=augmenter, seed=cfg.data.seed)
    val_ds = make_dataset(val_df, cfg.data.image_size, batch_size, training=False)
    test_ds = make_dataset(test_df, cfg.data.image_size, batch_size, training=False, cache=False)

    from .models.efficientnet import build_efficientnet, unfreeze_top

    ecfg = cfg.efficientnet
    tcfg = cfg.train
    out_dir = C.MODELS_DIR / "efficientnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = C.PLOTS_DIR / "efficientnet"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model = build_efficientnet(len(class_names), (*cfg.data.image_size, 3), ecfg.dropout)
    total_params = int(model.count_params())
    log.info("Model: %d total params, backbone frozen for Stage A", total_params)

    # ---- Stage A: frozen backbone, train head only ----
    log.info("=" * 60)
    log.info("STAGE A: Feature extraction (frozen backbone) | %d epochs | lr=%g", ecfg.head_epochs, ecfg.head_learning_rate)
    log.info("=" * 60)

    model.compile(optimizer=keras.optimizers.Adam(ecfg.head_learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cbA = [
        keras.callbacks.ModelCheckpoint(str(out_dir / "stageA_best.keras"), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=ecfg.early_stopping_patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=tcfg.reduce_lr_factor, patience=tcfg.reduce_lr_patience, min_lr=tcfg.min_lr, verbose=1),
        keras.callbacks.CSVLogger(str(out_dir / "stageA_log.csv")),
    ]

    t0 = time.perf_counter()
    histA = model.fit(train_ds, validation_data=val_ds, epochs=ecfg.head_epochs, class_weight=cw, verbose=2, callbacks=cbA)
    tA = time.perf_counter() - t0
    hA = {k: [float(v) for v in vs] for k, vs in histA.history.items()}
    save_json(hA, out_dir / "stageA_history.json")

    from .plots import plot_history
    plot_history(hA, plots_dir / "stageA", "EfficientNetB0 - Stage A (head only)")
    model.save(out_dir / "stageA_model.keras")
    beA = int(np.argmin(hA["val_loss"])) + 1
    log.info("Stage A: %d epochs in %.1fs, best ep %d (val_loss=%.4f)", len(hA["loss"]), tA, beA, min(hA["val_loss"]))

    # ---- Stage B: fine-tune from block5 ----
    log.info("=" * 60)
    log.info("STAGE B: Fine-tuning from '%s' | %d epochs | lr=%g", ecfg.finetune_unfreeze_from, ecfg.finetune_epochs, ecfg.finetune_learning_rate)
    log.info("=" * 60)

    n_unfrozen = unfreeze_top(model, ecfg.finetune_unfreeze_from)
    trainable_b = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    log.info("Fine-tuning: %d backbone layers unfrozen, %d trainable params", n_unfrozen, trainable_b)

    model.compile(optimizer=keras.optimizers.Adam(ecfg.finetune_learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cbB = [
        keras.callbacks.ModelCheckpoint(str(out_dir / "stageB_best.keras"), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=ecfg.early_stopping_patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=tcfg.reduce_lr_factor, patience=tcfg.reduce_lr_patience, min_lr=tcfg.min_lr, verbose=1),
        keras.callbacks.CSVLogger(str(out_dir / "stageB_log.csv")),
    ]

    t1 = time.perf_counter()
    histB = model.fit(train_ds, validation_data=val_ds, epochs=ecfg.finetune_epochs, class_weight=cw, verbose=2, callbacks=cbB)
    tB = time.perf_counter() - t1
    hB = {k: [float(v) for v in vs] for k, vs in histB.history.items()}
    save_json(hB, out_dir / "stageB_history.json")
    plot_history(hB, plots_dir / "stageB", "EfficientNetB0 - Stage B (fine-tuning)")

    combined = {k: hA[k] + hB[k] for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
    plot_history(combined, plots_dir / "combined", "EfficientNetB0 (Stage A + B)", stage_boundary=len(hA["loss"]))
    model.save(out_dir / "stageB_model.keras")
    beB = int(np.argmin(hB["val_loss"])) + 1
    log.info("Stage B: %d epochs in %.1fs, best ep %d (val_loss=%.4f)", len(hB["loss"]), tB, beB, min(hB["val_loss"]))

    # ---- Select best stage on validation ----
    from .evaluate import compute_metrics, predict_dataset

    modelA = keras.models.load_model(out_dir / "stageA_model.keras")
    y_true_v, y_prob_vA = predict_dataset(modelA, val_ds)
    mA = compute_metrics(y_true_v, y_prob_vA.argmax(1), class_names)

    _, y_prob_vB = predict_dataset(model, val_ds)
    mB = compute_metrics(y_true_v, y_prob_vB.argmax(1), class_names)

    stages = {
        "stageA_feature_extraction": {"val_metrics": mA, "val_loss": min(hA["val_loss"]), "epochs_run": len(hA["loss"]), "best_epoch": beA, "train_time_s": tA},
        "stageB_fine_tuning": {"val_metrics": mB, "val_loss": min(hB["val_loss"]), "epochs_run": len(hB["loss"]), "best_epoch": beB, "train_time_s": tB, "unfrozen_layers": n_unfrozen},
    }

    log.info("Val: Stage A macro-F1=%.4f | Stage B macro-F1=%.4f", mA["macro_f1"], mB["macro_f1"])
    sel = max(stages, key=lambda s: (round(stages[s]["val_metrics"]["macro_f1"], 4), -stages[s]["val_loss"]))
    log.info("Selected: %s (val macro-F1=%.4f)", sel, stages[sel]["val_metrics"]["macro_f1"])

    import shutil
    src_ckpt = out_dir / ("stageB_model.keras" if sel == "stageB_fine_tuning" else "stageA_model.keras")
    shutil.copyfile(src_ckpt, out_dir / "model.keras")

    total_time = tA + tB
    save_json({"class_names": class_names, "class_to_index": mapping}, out_dir / "class_mapping.json")
    save_json(preprocessing_spec("efficientnet", cfg.data.image_size), out_dir / "preprocessing.json")
    save_json({"model": "efficientnet", "selected_stage": sel, "train_time_s": total_time,
               "params": total_params, "train_size": len(train_df), "val_size": len(val_df),
               "smoke": smoke, "stages": stages, "config": cfg.to_dict()},
              out_dir / "training_summary.json")
    log.info("Training artifacts saved to %s", out_dir)

    # ---- Evaluate on held-out test set ----
    log.info("=" * 60)
    log.info("TEST SET EVALUATION (never used during training or model selection)")
    log.info("=" * 60)

    best_model = keras.models.load_model(out_dir / "model.keras")
    y_true_t, y_prob_t = predict_dataset(best_model, test_ds)
    y_pred_t = y_prob_t.argmax(1)
    test_m = compute_metrics(y_true_t, y_pred_t, class_names)

    from .plots import plot_confusion_matrix
    import pandas as pd

    out_metrics = C.METRICS_DIR / "efficientnet"
    out_metrics.mkdir(parents=True, exist_ok=True)

    save_json({k: v for k, v in test_m.items() if k != "classification_report"}, out_metrics / "test_metrics.json")
    (out_metrics / "test_classification_report.txt").write_text(test_m["classification_report"], encoding="utf-8")
    pd.DataFrame({"filepath": test_df["filepath"], "true": [class_names[i] for i in y_true_t],
                  "pred": [class_names[i] for i in y_pred_t], "confidence": y_prob_t.max(1),
                  **{f"p_{c}": y_prob_t[:, i] for i, c in enumerate(class_names)}}).to_csv(
        out_metrics / "test_predictions.csv", index=False)

    plot_confusion_matrix(np.array(test_m["confusion_matrix"]), class_names,
                          plots_dir / "test_confusion_matrix.png",
                          f"EfficientNetB0 - Test Confusion Matrix (n={len(y_true_t)})")

    log.info("\n%s", test_m["classification_report"])
    log.info("TEST: Accuracy=%.4f | Macro F1=%.4f | Weighted F1=%.4f",
             test_m["accuracy"], test_m["macro_f1"], test_m["weighted_f1"])

    # Misclassification analysis
    preds_df = pd.read_csv(out_metrics / "test_predictions.csv")
    wrong = preds_df[preds_df["true"] != preds_df["pred"]]
    log.info("Errors: %d / %d (%.1f%% error rate)", len(wrong), len(preds_df), 100*len(wrong)/len(preds_df))
    err_table = wrong.groupby(["true", "pred"]).size().rename("count").reset_index().sort_values("count", ascending=False)
    log.info("Most common errors:\n%s", err_table.head(10).to_string(index=False))
    err_table.to_csv(out_metrics / "test_error_analysis.csv", index=False)

    return {"test_accuracy": test_m["accuracy"], "macro_f1": test_m["macro_f1"],
            "weighted_f1": test_m["weighted_f1"], "selected_stage": sel, "total_train_time_s": total_time}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="Tiny subset, 1 epoch per stage")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    main(smoke=args.smoke, batch_size=args.batch_size)
