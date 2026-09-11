"""Diagnóstico de una predicción individual CINC2020-12.

Sirve para responder si una mala predicción proviene de:
  1) el modelo/checkpoint;
  2) una diferencia entre preprocesamiento HDF5 y preprocesamiento de la webapp;
  3) umbrales/postprocesamiento.

Ejemplo:

python examples/cinc2020/debug_record_prediction.py \
  --config examples/cinc2020/config.json \
  --checkpoint saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --record E00001 \
  --hea dataset2020/training/georgia/g1/E00001.hea \
  --mat dataset2020/training/georgia/g1/E00001.mat \
  --thresholds results/cinc2020_12_resnet/thresholds_validation.csv \
  --normal-fallback-min-prob 0.40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg import load, predict, util
from webapp.prediction import (
    apply_normal_fallback,
    build_label_comparison,
    threshold_values_for_class_names,
)


def read_thresholds(path: str | None, class_names: list[str]) -> dict | None:
    if not path:
        return None
    p = util.resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe thresholds: {p}")
    df = pd.read_csv(p)
    if "class" not in df.columns or "threshold" not in df.columns:
        raise ValueError("thresholds_validation.csv debe tener columnas class y threshold")
    values = {str(row["class"]): float(row["threshold"]) for _, row in df.iterrows()}
    missing = [name for name in class_names if name not in values]
    if missing:
        raise ValueError(f"Faltan thresholds para: {missing}")
    return values


def find_record_in_hdf5(config: dict, record_name: str):
    import h5py

    candidates = [
        ("train", config.get("train")),
        ("validation", config.get("dev")),
        ("test", config.get("test")),
    ]
    target = record_name.lower()
    for split, h5_path in candidates:
        if not h5_path:
            continue
        path = util.resolve_path(h5_path)
        if not path.exists():
            continue
        with h5py.File(path, "r") as h5:
            names = h5.get("record_names")
            if names is None:
                continue
            decoded = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in names[:]]
            h5_classes = h5.attrs.get("classes", None)
            if h5_classes is not None:
                h5_classes = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in h5_classes]
            else:
                h5_classes = load.CLASS_NAMES
            for i, name in enumerate(decoded):
                if name.lower() == target:
                    signal = np.asarray(h5["signals"][i], dtype=np.float32)
                    label = np.asarray(h5["labels"][i], dtype=np.uint8)
                    return {
                        "split": split,
                        "path": path,
                        "index": i,
                        "signal": signal,
                        "label": label,
                        "h5_classes": list(h5_classes),
                        "true_classes": [h5_classes[j] for j, active in enumerate(label) if active],
                    }
    return None


def rows_from_probabilities(probabilities, class_names, thresholds, fallback_min_prob):
    rows = []
    for i, prob in enumerate(probabilities):
        thr = float(thresholds[i])
        rows.append({
            "index": i,
            "class": class_names[i],
            "probability": float(prob),
            "threshold": thr,
            "prediction": int(float(prob) >= thr),
        })
    raw = sorted([dict(r) for r in rows if r["prediction"]], key=lambda r: r["probability"], reverse=True)
    fallback = apply_normal_fallback(rows, min_nsr_probability=fallback_min_prob) if fallback_min_prob and fallback_min_prob > 0 else {"applied": False}
    rows_sorted = sorted(rows, key=lambda r: r["probability"], reverse=True)
    final = [r for r in rows_sorted if r["prediction"]]
    return rows_sorted, raw, final, fallback


def print_table(title, rows_sorted, max_rows=12):
    print("\n" + title)
    print("-" * len(title))
    print(f"{'rank':>4} {'class':<20} {'prob':>9} {'thr':>9} {'margin':>9} {'pred':>6} {'post':>6}")
    for rank, row in enumerate(rows_sorted[:max_rows], 1):
        prob = float(row["probability"])
        thr = float(row["threshold"])
        print(f"{rank:4d} {row['class']:<20} {prob:9.4f} {thr:9.4f} {prob-thr:9.4f} {int(row['prediction']):6d} {str(bool(row.get('postprocessed', False))):>6}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Depura una predicción CINC2020-12 por registro")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--record", required=True, help="record_name, por ejemplo E00001")
    parser.add_argument("--hea", default=None)
    parser.add_argument("--mat", default=None)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--normal-fallback-min-prob", type=float, default=0.40)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args(argv)

    config = json.loads(util.resolve_path(args.config).read_text(encoding="utf-8"))
    device = predict.get_device(args.device)
    model, checkpoint, class_names = predict.load_model(args.checkpoint, device=device)
    pre = predict.preprocessing_for_checkpoint(checkpoint)
    thresholds_dict = read_thresholds(args.thresholds, class_names)
    thresholds = threshold_values_for_class_names(class_names, threshold=0.5, thresholds=thresholds_dict)

    print("=" * 78)
    print("DEBUG PREDICCIÓN CINC2020-12")
    print("=" * 78)
    print("Registro     :", args.record)
    print("Checkpoint   :", util.resolve_path(args.checkpoint))
    print("Esquema ckpt :", checkpoint.get("label_schema"), "| norm:", pre["norm_mode"], "| bandpass:", pre["bandpass"])
    print("Época/val_loss:", checkpoint.get("epoch"), checkpoint.get("val_loss"))
    print("Device       :", device)
    print("Thresholds   :", args.thresholds or "fallback global 0.5")
    print("Fallback NSR :", args.normal_fallback_min_prob)

    h5_item = find_record_in_hdf5(config, args.record)
    wfdb_signal = None

    if args.hea and args.mat:
        wfdb_signal, meta = load.load_wfdb_record(
            args.mat, args.hea, norm_mode=pre["norm_mode"], bandpass=pre["bandpass"])
        true_classes = load.matched_class_names(meta.get("dx_codes", []))
        if "LAD" in class_names and "AxisDev" not in class_names:
            true_classes = [{"AxisDev": "LAD", "BBB": "RBBB"}.get(c, c) for c in true_classes]
        prob = predict.predict_array(model, wfdb_signal, device=device)
        rows, raw_pos, final_pos, fallback = rows_from_probabilities(prob, class_names, thresholds, args.normal_fallback_min_prob)
        print_table("WEBAPP/WFDB preprocesado", rows)
        cmp = build_label_comparison(true_classes, rows, "wfdb_header_dx", dx_codes=meta.get("dx_codes", []))
        print("\nEtiqueta header:", true_classes or "Ninguna de las 12")
        print("Raw positives :", [r["class"] for r in raw_pos] or "Ninguna")
        print("Final positives:", [r["class"] for r in final_pos] or "Ninguna")
        print("Fallback      :", fallback)
        print("Exact match   :", cmp.get("exact_match"), "F1=", cmp.get("f1"), "Jaccard=", cmp.get("jaccard"))

    if h5_item is not None:
        print(f"\nHDF5 {h5_item['path']} clases={h5_item.get('h5_classes')}")
        prob = predict.predict_array(model, h5_item["signal"], device=device)
        rows, raw_pos, final_pos, fallback = rows_from_probabilities(prob, class_names, thresholds, args.normal_fallback_min_prob)
        print_table(f"HDF5 {h5_item['split']} index={h5_item['index']}", rows)
        cmp = build_label_comparison(h5_item["true_classes"], rows, "hdf5_label")
        print("\nEtiqueta HDF5 :", h5_item["true_classes"] or "Ninguna")
        print("Raw positives :", [r["class"] for r in raw_pos] or "Ninguna")
        print("Final positives:", [r["class"] for r in final_pos] or "Ninguna")
        print("Fallback      :", fallback)
        print("Exact match   :", cmp.get("exact_match"), "F1=", cmp.get("f1"), "Jaccard=", cmp.get("jaccard"))
    else:
        print("\nRegistro no encontrado en HDF5 train/validation/test configurados.")

    if wfdb_signal is not None and h5_item is not None:
        diff = np.asarray(wfdb_signal) - np.asarray(h5_item["signal"])
        print("\nComparación señal webapp vs HDF5")
        print("  max_abs:", float(np.max(np.abs(diff))))
        print("  mean_abs:", float(np.mean(np.abs(diff))))
        print("  shape  :", wfdb_signal.shape, h5_item["signal"].shape)
        if float(np.max(np.abs(diff))) < 1e-5:
            print("  Conclusión: el preprocesamiento coincide; el error viene del modelo/umbral/postproceso.")
        else:
            print("  Conclusión: hay diferencia de preprocesamiento; revisar lectura/remuestreo/normalización.")


if __name__ == "__main__":
    main()
