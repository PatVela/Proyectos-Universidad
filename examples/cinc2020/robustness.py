"""Experimento reproducible de robustez para CINC2020-12.

Evalúa el mismo checkpoint bajo perturbaciones controladas sobre el HDF5 test:
ruido gaussiano, baseline wander, escalado de amplitud y apagado de derivaciones.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from ecg import predict, util


def apply_perturbation(x, kind, level, rng):
    """x: tensor (B, 12, 5000) ya normalizado."""
    if kind == "clean":
        return x
    if kind == "gaussian_noise":
        return x + torch.randn_like(x) * float(level)
    if kind == "amplitude_scale":
        return x * float(level)
    if kind == "baseline_wander":
        b, c, n = x.shape
        t = torch.linspace(0, 1, n, device=x.device, dtype=x.dtype)[None, None, :]
        phase = torch.tensor(rng.uniform(0, 2 * np.pi, size=(b, c, 1)), device=x.device, dtype=x.dtype)
        wander = float(level) * torch.sin(2 * np.pi * 0.5 * t + phase)
        return x + wander
    if kind == "lead_dropout":
        out = x.clone()
        num_drop = max(1, int(round(float(level))))
        for i in range(out.shape[0]):
            leads = rng.choice(out.shape[1], size=min(num_drop, out.shape[1]), replace=False)
            out[i, leads, :] = 0
        return out
    raise ValueError(f"Perturbación desconocida: {kind}")


@torch.no_grad()
def evaluate_under_perturbation(model, dataset, thresholds, device, kind, level, batch_size, num_workers, max_samples, seed):
    rng = np.random.default_rng(seed)
    if max_samples is not None and max_samples < len(dataset):
        indices = rng.choice(len(dataset), size=max_samples, replace=False)
        dataset = Subset(dataset, indices.tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    probs, labels = [], []
    for x, y in tqdm(loader, desc=f"{kind}:{level}", leave=False):
        x = x.to(device, non_blocking=True)
        x = apply_perturbation(x, kind, level, rng)
        p = torch.sigmoid(model(x)).float().cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0).astype(np.uint8)
    y_pred = (y_prob >= thresholds[None, :]).astype(np.uint8)
    return {
        "perturbation": kind,
        "level": level,
        "samples": int(y_true.shape[0]),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def load_thresholds(path, class_names):
    if not path:
        return np.full(len(class_names), 0.5, dtype=np.float32)
    df = pd.read_csv(path)
    values = np.full(len(class_names), 0.5, dtype=np.float32)
    for i, name in enumerate(class_names):
        row = df[df["class"] == name]
        if not row.empty:
            values[i] = float(row.iloc[0]["threshold"])
    return values


def main():
    parser = argparse.ArgumentParser(description="Robustez controlada CINC2020-12")
    parser.add_argument("checkpoint")
    parser.add_argument("test_h5", default="data/cinc2020_12/test.h5")
    parser.add_argument("--thresholds", default=None, help="thresholds_validation.csv; si se omite usa 0.5")
    parser.add_argument("--output", default="results/cinc2020_12/robustness.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    util.set_seed(args.seed)
    device = predict.get_device(args.device)
    model, _checkpoint, class_names = predict.load_model(args.checkpoint, device=device)
    dataset = predict.HDF5PredictionDataset(args.test_h5)
    thresholds = load_thresholds(args.thresholds, class_names)

    perturbations = [
        ("clean", 0.0),
        ("gaussian_noise", 0.05),
        ("gaussian_noise", 0.10),
        ("baseline_wander", 0.10),
        ("baseline_wander", 0.20),
        ("amplitude_scale", 0.5),
        ("amplitude_scale", 1.5),
        ("lead_dropout", 1),
        ("lead_dropout", 3),
    ]

    rows = []
    for kind, level in perturbations:
        rows.append(evaluate_under_perturbation(
            model, dataset, thresholds, device, kind, level,
            args.batch_size, args.num_workers, args.max_samples, args.seed,
        ))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nRobustez guardada en: {out}")


if __name__ == "__main__":
    main()
