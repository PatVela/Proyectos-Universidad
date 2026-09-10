"""Genera un HDF5 sintético pequeño para smoke tests.

Mantiene el nombre `syntethic` para coincidir con el archivo de configuración.
No se debe usar para reportar métricas científicas.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from ecg import load


def write_split(path: Path, n: int, seed: int):
    import h5py

    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    signals = rng.normal(0, 1, size=(n, load.WINDOW_LENGTH, load.NUM_LEADS)).astype(np.float32)
    labels = np.zeros((n, load.NUM_CLASSES), dtype=np.float32)
    for i in range(n):
        labels[i, i % load.NUM_CLASSES] = 1
        if i % 5 == 0:
            labels[i, (i + 3) % load.NUM_CLASSES] = 1
        if i % 11 == 0:
            labels[i, :] = 0  # algunos all-zero

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("signals", data=signals, compression="gzip", compression_opts=2)
        h5.create_dataset("labels", data=labels)
        h5.create_dataset("ages", data=rng.integers(18, 90, size=n).astype(np.float32))
        h5.create_dataset("sexes", data=np.asarray(["Unknown"] * n, dtype=object), dtype=string_dtype)
        h5.create_dataset("source_dbs", data=np.asarray(["synthetic"] * n, dtype=object), dtype=string_dtype)
        h5.create_dataset("record_names", data=np.asarray([f"synth_{i:05d}" for i in range(n)], dtype=object), dtype=string_dtype)
        h5.attrs["dataset"] = "Synthetic smoke-test"
        h5.attrs["label_schema"] = "cinc2020_12_grouped_snomed"
        h5.attrs["label_mode"] = "multilabel"
        h5.attrs["sampling_rate"] = load.TARGET_FS
        h5.attrs["window_length"] = load.WINDOW_LENGTH
        h5.attrs["num_leads"] = load.NUM_LEADS
        h5.attrs["num_classes"] = load.NUM_CLASSES
        h5.attrs["lead_order"] = ",".join(load.STANDARD_LEAD_ORDER)
        h5.attrs["normalization"] = "synthetic standard normal"
        h5.attrs.create("classes", np.asarray(load.CLASS_NAMES, dtype=string_dtype))


def main():
    parser = argparse.ArgumentParser(description="Genera dataset sintético mínimo CINC2020-12")
    parser.add_argument("--output-dir", default="data/cinc2020_synth")
    parser.add_argument("--train", type=int, default=48)
    parser.add_argument("--val", type=int, default=24)
    parser.add_argument("--test", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    write_split(out / "train.h5", args.train, args.seed)
    write_split(out / "val.h5", args.val, args.seed + 1)
    write_split(out / "test.h5", args.test, args.seed + 2)
    print(f"Dataset sintético escrito en: {out}")


if __name__ == "__main__":
    main()
