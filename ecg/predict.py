"""Inferencia desde checkpoints CINC2020-12.

Uso CSV::

    python -m ecg.predict saved/cinc2020/cinc2020_resnet/<run>/best.pt ecg.csv

El CSV debe tener 12 columnas de derivaciones. Opcionalmente puede comenzar con
una línea ``# Sampling Rate: 500 Hz``. La salida del modelo es multilabel: se
aplica sigmoid a 12 logits independientes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from . import load, network, util
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from ecg import load, network, util


class HDF5PredictionDataset(Dataset):
    """Dataset HDF5 mínimo para inferencia/evaluación."""

    def __init__(self, h5_path: str | Path):
        import h5py

        self.h5_path = str(util.resolve_path(h5_path))
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"No existe HDF5: {self.h5_path}")
        with h5py.File(self.h5_path, "r") as h5:
            self.length = int(h5["signals"].shape[0])
            self.signal_shape = tuple(h5["signals"].shape)
            self.label_shape = tuple(h5["labels"].shape) if "labels" in h5 else None
            classes = h5.attrs.get("classes", None)
            self.classes = [_decode(c) for c in classes] if classes is not None else load.CLASS_NAMES.copy()
            if "record_names" in h5:
                self.record_names = [_decode(x) for x in h5["record_names"][:]]
            else:
                self.record_names = [f"record_{i}" for i in range(self.length)]
        self.h5 = None

    def _open(self):
        if self.h5 is None:
            import h5py
            self.h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._open()
        x = np.asarray(self.h5["signals"][index], dtype=np.float32).T
        if "labels" in self.h5:
            y = np.asarray(self.h5["labels"][index], dtype=np.float32)
        else:
            y = np.zeros(load.NUM_CLASSES, dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __del__(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_params_from_config(config: dict) -> dict:
    return {
        "num_leads": int(config.get("num_leads", load.NUM_LEADS)),
        "num_classes": int(config.get("num_classes", load.NUM_CLASSES)),
        "input_length": int(config.get("input_length", load.WINDOW_LENGTH)),
        "conv_filter_length": int(config.get("conv_filter_length", config.get("kernel_size", 7))),
        "conv_num_filters_start": int(config.get("conv_num_filters_start", config.get("filter_length", 32))),
        "conv_subsample_lengths": config.get("conv_subsample_lengths", [1, 2] * 8),
        "conv_num_skip": int(config.get("conv_num_skip", 2)),
        "conv_dropout": float(config.get("conv_dropout", config.get("drop_rate", 0.2))),
        "is_regular_conv": bool(config.get("is_regular_conv", False)),
    }


def load_model(checkpoint_path: str | Path, device: torch.device | None = None):
    """Carga checkpoint y devuelve ``(model, checkpoint, class_names)``."""
    device = device or get_device("auto")
    checkpoint_path = util.resolve_path(checkpoint_path)
    checkpoint = util.load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    class_names = checkpoint.get("class_names", load.CLASS_NAMES.copy())
    if len(class_names) != load.NUM_CLASSES:
        raise ValueError(f"El checkpoint tiene {len(class_names)} clases; se esperaban {load.NUM_CLASSES}")

    params = model_params_from_config(config)
    params["num_classes"] = len(class_names)
    model = network.build_network(**params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, class_names


@torch.no_grad()
def predict_array(model, signal: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Predice probabilidades para una señal preprocesada ``(5000,12)`` o ``(12,5000)``."""
    device = device or next(model.parameters()).device
    x = np.asarray(signal, dtype=np.float32)
    if x.shape == (load.WINDOW_LENGTH, load.NUM_LEADS):
        x = x.T
    elif x.shape == (load.NUM_LEADS, load.WINDOW_LENGTH):
        pass
    else:
        raise ValueError(f"Forma no válida para inferencia: {x.shape}")
    tensor = torch.from_numpy(x[None, ...]).to(device)
    logits = model(tensor)
    return torch.sigmoid(logits)[0].detach().cpu().numpy()


def predict_csv(checkpoint_path: str | Path, csv_path: str | Path, threshold: float = 0.5, device: str = "auto") -> dict:
    """Lee un CSV ECG, preprocesa y predice 12 probabilidades."""
    torch_device = get_device(device)
    model, checkpoint, class_names = load_model(checkpoint_path, torch_device)
    raw, sampling_rate, lead_names = load.read_csv_ecg(csv_path)
    processed = load.preprocess_ecg_array(raw, sampling_rate=sampling_rate, lead_names=lead_names)
    probabilities = predict_array(model, processed, torch_device)
    predictions = (probabilities >= threshold).astype(np.uint8)
    rows = [
        {
            "index": i,
            "class": class_names[i],
            "probability": float(probabilities[i]),
            "prediction": int(predictions[i]),
        }
        for i in range(len(class_names))
    ]
    return {
        "checkpoint": str(checkpoint_path),
        "csv_path": str(csv_path),
        "threshold": float(threshold),
        "class_names": class_names,
        "probabilities": probabilities,
        "predictions": predictions,
        "rows": rows,
        "sampling_rate": sampling_rate,
        "lead_names": lead_names,
        "processed_shape": list(processed.shape),
    }


@torch.no_grad()
def predict_hdf5(
    checkpoint_path: str | Path,
    h5_path: str | Path,
    batch_size: int = 8,
    num_workers: int = 2,
    device: str = "auto",
    amp: bool = True,
):
    """Predice todo un HDF5. Devuelve probabilidades, etiquetas y metadata."""
    torch_device = get_device(device)
    model, checkpoint, class_names = load_model(checkpoint_path, torch_device)
    dataset = HDF5PredictionDataset(h5_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(torch_device.type == "cuda"))

    all_prob, all_true = [], []
    use_amp = amp and torch_device.type == "cuda"
    for x, y in tqdm(loader, desc=f"Prediciendo {Path(h5_path).name}", leave=False):
        x = x.to(torch_device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probs = torch.sigmoid(model(x))
        else:
            probs = torch.sigmoid(model(x))
        all_prob.append(probs.float().cpu().numpy())
        all_true.append(y.cpu().numpy())

    return {
        "probabilities": np.concatenate(all_prob, axis=0),
        "labels": np.concatenate(all_true, axis=0),
        "record_names": dataset.record_names,
        "class_names": class_names,
        "checkpoint": checkpoint,
    }


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inferencia CINC2020-12 desde checkpoint")
    parser.add_argument("checkpoint", help="Checkpoint .pt")
    parser.add_argument("input", help="CSV ECG o HDF5")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default=None, help="CSV/JSON de salida")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".h5":
        result = predict_hdf5(args.checkpoint, input_path, batch_size=args.batch_size, num_workers=args.num_workers, device=args.device)
        probs = result["probabilities"]
        payload = {"record_name": result["record_names"]}
        for i, name in enumerate(result["class_names"]):
            payload[f"prob_{name}"] = probs[:, i]
            payload[f"pred_{name}"] = (probs[:, i] >= args.threshold).astype(np.uint8)
        df = pd.DataFrame(payload)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Predicciones guardadas en {args.output}")
        else:
            print(df.head().to_string(index=False))
    else:
        result = predict_csv(args.checkpoint, input_path, threshold=args.threshold, device=args.device)
        df = rows_to_dataframe(result["rows"])
        if args.output:
            out = Path(args.output)
            if out.suffix.lower() == ".json":
                util.save_json(out, {k: v for k, v in result.items() if k not in {"probabilities", "predictions"}})
            else:
                df.to_csv(out, index=False)
            print(f"Predicción guardada en {out}")
        else:
            print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
