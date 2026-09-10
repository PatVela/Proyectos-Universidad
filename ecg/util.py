"""Utilidades compartidas: seeds, paths, checkpoints, conteo de parámetros."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int | None) -> None:
    """Fija semillas de Python, NumPy y PyTorch si está disponible."""
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
    except Exception:
        pass


def resolve_path(path: str | os.PathLike, base: str | os.PathLike | None = None) -> Path:
    """Resuelve rutas relativas respecto a la raíz del repositorio por defecto."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(base) if base is not None else REPO_ROOT) / path


def timestamped_dir(root: str | os.PathLike, experiment: str) -> Path:
    """Crea directorio reproducible estilo ``save_dir/experiment/timestamp-rand``."""
    root = resolve_path(root)
    dirname = f"{int(time.time())}-{random.randrange(1000):03d}"
    save_dir = root / experiment / dirname
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def count_parameters(model) -> dict:
    """Devuelve conteo total/entrenable/no entrenable de parámetros PyTorch."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(total - trainable),
    }


def save_json(path: str | os.PathLike, payload: dict) -> None:
    """Guarda JSON con conversión básica NumPy/Path."""
    def convert(obj: Any):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=convert), encoding="utf-8")


def load_json(path: str | os.PathLike) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(
    path: str | os.PathLike,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    val_loss: float | None = None,
    config: dict | None = None,
    class_names: list[str] | None = None,
    extra: dict | None = None,
) -> None:
    """Guarda checkpoint PyTorch completo."""
    import torch

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch) if epoch is not None else None,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "config": config or {},
        "class_names": class_names or [],
        "num_classes": len(class_names or []),
        "label_schema": "cinc2020_12_grouped_snomed",
        "problem_type": "multilabel_sigmoid_bce",
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    """Carga checkpoint PyTorch; ``weights_only=False`` por compatibilidad."""
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_checkpoints(models_dir: str | os.PathLike) -> list[Path]:
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    return sorted(models_dir.rglob("*.pt"))


def best_checkpoint(models_dir: str | os.PathLike) -> Path | None:
    """Busca checkpoint con menor ``val_loss`` interna o menor prefijo numérico."""
    checkpoints = find_checkpoints(models_dir)
    if not checkpoints:
        return None

    def score(path: Path):
        try:
            ckpt = load_checkpoint(path, map_location="cpu")
            value = ckpt.get("val_loss", None)
            if value is not None:
                return float(value)
        except Exception:
            pass
        try:
            return float(path.name.split("-")[0])
        except Exception:
            return float("inf")

    return min(checkpoints, key=score)
