"""Evaluación multilabel de un checkpoint CINC2020-12.

Optimiza thresholds por clase en validation y evalúa test con esos thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from ecg import load, predict, util


def safe_metric(fn, y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(fn(y_true, y_score))
    except Exception:
        return np.nan


def safe_macro(y_true, y_prob, fn):
    values = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        try:
            values.append(float(fn(y_true[:, i], y_prob[:, i])))
        except Exception:
            pass
    return float(np.mean(values)) if values else np.nan


def best_threshold_f1(y_true, y_prob):
    if y_true.sum() == 0:
        return 0.5, 0.0
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        score = f1_score(y_true, (y_prob >= t).astype(np.uint8), zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def optimize_thresholds(y_true, y_prob, class_names):
    thresholds = np.zeros(y_true.shape[1], dtype=np.float32)
    rows = []
    for i, name in enumerate(class_names):
        t, score = best_threshold_f1(y_true[:, i].astype(np.uint8), y_prob[:, i])
        thresholds[i] = t
        rows.append({
            "index": i,
            "class": name,
            "threshold": t,
            "validation_f1": score,
            "validation_positives": int(y_true[:, i].sum()),
        })
    return thresholds, pd.DataFrame(rows)


def apply_normal_fallback(y_pred, y_prob, class_names, min_prob=None):
    """Añade NSR cuando ninguna clase supera umbral y P(NSR) es razonable."""
    if min_prob is None or float(min_prob) <= 0 or "NSR" not in class_names:
        return y_pred, 0
    y_pred = y_pred.copy()
    nsr_idx = list(class_names).index("NSR")
    empty = y_pred.sum(axis=1) == 0
    mask = empty & (y_prob[:, nsr_idx] >= float(min_prob))
    y_pred[mask, nsr_idx] = 1
    return y_pred, int(mask.sum())


def evaluate_predictions(y_true, y_prob, thresholds, class_names, split, normal_fallback_min_prob=None):
    y_true = y_true.astype(np.uint8)
    y_pred = (y_prob >= thresholds[None, :]).astype(np.uint8)
    y_pred, normal_fallback_count = apply_normal_fallback(y_pred, y_prob, class_names, normal_fallback_min_prob)
    rows = []
    for i, name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
        rows.append({
            "split": split,
            "index": i,
            "class": name,
            "threshold": float(thresholds[i]),
            "support_positive": int(y_true[:, i].sum()),
            "support_negative": int(y_true.shape[0] - y_true[:, i].sum()),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "sensitivity_recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "specificity": specificity,
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "auroc": safe_metric(roc_auc_score, y_true[:, i], y_prob[:, i]),
            "auprc": safe_metric(average_precision_score, y_true[:, i], y_prob[:, i]),
        })
    class_df = pd.DataFrame(rows)
    global_metrics = {
        "split": split,
        "samples": int(y_true.shape[0]),
        "classes": int(y_true.shape[1]),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "sensitivity_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "specificity_macro": float(class_df["specificity"].mean()),
        "auroc_macro": safe_macro(y_true, y_prob, roc_auc_score),
        "auprc_macro": safe_macro(y_true, y_prob, average_precision_score),
        "exact_match_ratio": float((y_true == y_pred).all(axis=1).mean()),
        "normal_fallback_min_prob": None if normal_fallback_min_prob is None else float(normal_fallback_min_prob),
        "normal_fallback_count": int(normal_fallback_count),
    }
    return class_df, global_metrics, y_pred


def save_predictions(path, record_names, y_true, y_prob, y_pred, class_names):
    payload = {"record_name": record_names}
    for i, name in enumerate(class_names):
        payload[f"true_{name}"] = y_true[:, i].astype(np.uint8)
        payload[f"prob_{name}"] = y_prob[:, i].astype(np.float32)
        payload[f"pred_{name}"] = y_pred[:, i].astype(np.uint8)
    pd.DataFrame(payload).to_csv(path, index=False)


def save_confusion_matrices(out_dir, class_df, split):
    cm_dir = Path(out_dir) / "confusion_matrices" / split
    cm_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in class_df.iterrows():
        matrix = pd.DataFrame(
            [[int(row["TN"]), int(row["FP"])], [int(row["FN"]), int(row["TP"])]],
            index=["true_0", "true_1"],
            columns=["pred_0", "pred_1"],
        )
        safe = str(row["class"]).replace("/", "_").replace(" ", "_")
        matrix.to_csv(cm_dir / f"{int(row['index']):02d}_{safe}.csv")
        rows.append({"split": split, "class": row["class"], "TN": int(row["TN"]), "FP": int(row["FP"]), "FN": int(row["FN"]), "TP": int(row["TP"])})
    pd.DataFrame(rows).to_csv(Path(out_dir) / f"confusion_matrices_{split}.csv", index=False)


def run_evaluation(config, checkpoint, output_dir, batch_size=None, num_workers=None, device="auto", no_amp=False, normal_fallback_min_prob=None, preload=True):
    output_dir = util.resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_h5 = config["dev"]
    test_h5 = config["test"]
    batch_size = int(batch_size or config.get("batch_size", 8))
    num_workers = int(num_workers if num_workers is not None else config.get("num_workers", 2))

    val = predict.predict_hdf5(checkpoint, val_h5, batch_size=batch_size, num_workers=num_workers, device=device, amp=not no_amp, preload=preload)
    thresholds, threshold_df = optimize_thresholds(val["labels"], val["probabilities"], val["class_names"])
    threshold_df.to_csv(output_dir / "thresholds_validation.csv", index=False)

    val_class_df, val_global, val_pred = evaluate_predictions(
        val["labels"], val["probabilities"], thresholds, val["class_names"], "validation",
        normal_fallback_min_prob=normal_fallback_min_prob,
    )

    test = predict.predict_hdf5(checkpoint, test_h5, batch_size=batch_size, num_workers=num_workers, device=device, amp=not no_amp, preload=preload)
    test_class_df, test_global, test_pred = evaluate_predictions(
        test["labels"], test["probabilities"], thresholds, test["class_names"], "test",
        normal_fallback_min_prob=normal_fallback_min_prob,
    )

    pd.concat([val_class_df, test_class_df], ignore_index=True).to_csv(output_dir / "metrics_per_class.csv", index=False)
    pd.DataFrame([val_global, test_global]).to_csv(output_dir / "metrics_global.csv", index=False)
    save_predictions(output_dir / "predictions_validation.csv", val["record_names"], val["labels"], val["probabilities"], val_pred, val["class_names"])
    save_predictions(output_dir / "predictions_test.csv", test["record_names"], test["labels"], test["probabilities"], test_pred, test["class_names"])
    save_confusion_matrices(output_dir, val_class_df, "validation")
    save_confusion_matrices(output_dir, test_class_df, "test")

    summary = {
        "checkpoint": str(checkpoint),
        "label_schema": load.LABEL_SCHEMA,
        "problem_type": "multilabel sigmoid + BCEWithLogitsLoss",
        "class_names": val["class_names"],
        "thresholds": thresholds,
        "normal_fallback_min_prob": None if normal_fallback_min_prob is None else float(normal_fallback_min_prob),
        "validation": val_global,
        "test": test_global,
    }
    util.save_json(output_dir / "evaluation_summary.json", summary)
    print("\nTEST")
    print(pd.DataFrame([test_global]).to_string(index=False))
    print("\nMétricas por clase")
    print(test_class_df[["index", "class", "support_positive", "sensitivity_recall", "specificity", "f1", "auroc", "auprc"]].to_string(index=False))
    print(f"\nResultados guardados en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evalúa un checkpoint CINC2020-12")
    parser.add_argument("config", help="JSON de configuración")
    parser.add_argument("checkpoint", nargs="?", default=None, help="Checkpoint .pt; si se omite busca best.pt en save_dir")
    parser.add_argument("--output-dir", default="results/cinc2020_12")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-preload", action="store_true", help="Desactiva la precarga del HDF5 a RAM (lectura por disco, lento)")
    parser.add_argument("--normal-fallback-min-prob", type=float, default=None, help="Postprocesamiento opcional: si no hay positivos, añadir NSR cuando P(NSR) >= valor. Ejemplo: 0.40")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    checkpoint = args.checkpoint
    if checkpoint is None:
        best = util.best_checkpoint(util.resolve_path(config.get("save_dir", "saved/cinc2020")))
        if best is None:
            raise FileNotFoundError("No se indicó checkpoint y no se encontró ninguno en save_dir")
        checkpoint = best

    run_evaluation(
        config,
        checkpoint,
        args.output_dir,
        args.batch_size,
        args.num_workers,
        args.device,
        args.no_amp,
        normal_fallback_min_prob=args.normal_fallback_min_prob,
        preload=not args.no_preload,
    )


if __name__ == "__main__":
    main()
