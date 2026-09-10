"""Servicio de inferencia, conversión WFDB->CSV y gráficos para la webapp.

Entradas soportadas por la app:
    - CSV de 12 derivaciones.
    - Par WFDB del Challenge: archivo .hea + archivo .mat.

Si se sube .hea + .mat, la señal se lee, se preprocesa a 500 Hz / 5000 muestras
/ 12 derivaciones y se guarda además un CSV convertido para descarga.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

from ecg import load, util
from ecg.predict import load_model, predict_array


REPO_ROOT = Path(__file__).resolve().parent.parent
LEADS = load.STANDARD_LEAD_ORDER


# ---------------------------------------------------------------------------
# Rutas / modelo
# ---------------------------------------------------------------------------
def clean_path_text(value: str | os.PathLike | None) -> str:
    """Limpia rutas pegadas desde markdown, por ejemplo [best.pt](http://...)."""
    if value is None:
        return ""
    text = str(value).strip().strip('"').strip("'")
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return text.strip()


def resolve_existing_path(path: str | os.PathLike | None, base: Path = REPO_ROOT) -> Path | None:
    text = clean_path_text(path)
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def resolve_model_path(model_path: str | None = None, saved_dir: str | None = None) -> Path:
    """Resuelve checkpoint explícito o busca automáticamente en saved_dir."""
    explicit = resolve_existing_path(model_path)
    if explicit is not None and explicit.exists():
        return explicit

    saved_text = clean_path_text(saved_dir) or os.environ.get("ECG_SAVED", "saved")
    saved = Path(saved_text)
    if not saved.is_absolute():
        saved = REPO_ROOT / saved

    best = util.best_checkpoint(saved)
    if best is not None and best.exists():
        return best.resolve()

    message = "No se encontró un checkpoint .pt válido."
    if model_path:
        message += f" Checkpoint solicitado: {explicit}."
    message += f" Directorio de búsqueda: {saved}."
    raise FileNotFoundError(message)


# ---------------------------------------------------------------------------
# Preparación de entradas
# ---------------------------------------------------------------------------
def save_uploaded_files(file_storages: Iterable, upload_dir: str | Path) -> list[Path]:
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for storage in file_storages:
        if storage is None or not getattr(storage, "filename", ""):
            continue
        filename = secure_filename(storage.filename)
        if not filename:
            continue
        path = upload_dir / filename
        storage.save(path)
        saved_paths.append(path)

    if not saved_paths:
        raise ValueError("No se recibió ningún archivo válido.")
    return saved_paths


def _find_file_by_suffix(paths: Iterable[Path], suffix: str) -> Path | None:
    suffix = suffix.lower()
    matches = [path for path in paths if path.suffix.lower() == suffix]
    if not matches:
        return None
    return matches[0]


def write_processed_csv(signal: np.ndarray, output_path: str | Path) -> Path:
    """Guarda señal preprocesada (5000,12) como CSV descargable."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(signal, columns=LEADS)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# Sampling Rate: {load.TARGET_FS} Hz\n")
        df.to_csv(handle, index=False)
    return output_path


def prepare_saved_input(saved_paths: list[Path], result_dir: str | Path) -> dict:
    """Devuelve señal preprocesada y metadata desde CSV o par .hea+.mat."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _find_file_by_suffix(saved_paths, ".csv")
    hea_path = _find_file_by_suffix(saved_paths, ".hea")
    mat_path = _find_file_by_suffix(saved_paths, ".mat")

    if csv_path is not None and len(saved_paths) == 1:
        raw, sampling_rate, lead_names = load.read_csv_ecg(csv_path)
        processed = load.preprocess_ecg_array(raw, sampling_rate=sampling_rate, lead_names=lead_names)
        return {
            "input_type": "csv",
            "record_name": csv_path.stem,
            "source_files": [str(csv_path)],
            "csv_path": csv_path,
            "converted_csv_path": None,
            "processed_signal": processed,
            "original_sampling_rate": sampling_rate,
            "lead_names": lead_names,
            "dx_codes": [],
            "matched_classes": [],
            "true_label_source": None,
        }

    if hea_path is not None and mat_path is not None and len(saved_paths) == 2:
        if hea_path.stem != mat_path.stem:
            raise ValueError(
                "El archivo .hea y el archivo .mat deben pertenecer al mismo registro "
                f"(recibidos: {hea_path.name} y {mat_path.name})."
            )
        processed, meta = load.load_wfdb_record(mat_path, hea_path)
        record_name = meta.get("record_name") or hea_path.stem or mat_path.stem
        converted_csv = write_processed_csv(processed, result_dir / f"{secure_filename(record_name)}_convertido.csv")
        dx_codes = meta.get("dx_codes", [])
        return {
            "input_type": "wfdb_hea_mat",
            "record_name": record_name,
            "source_files": [str(hea_path), str(mat_path)],
            "csv_path": converted_csv,
            "converted_csv_path": converted_csv,
            "processed_signal": processed,
            "original_sampling_rate": meta.get("sampling_freq", load.TARGET_FS),
            "lead_names": meta.get("lead_names", LEADS),
            "dx_codes": dx_codes,
            "matched_classes": load.matched_class_names(dx_codes),
            "true_label_source": "wfdb_header_dx",
            "age": meta.get("age"),
            "sex": meta.get("sex"),
        }

    suffixes = ", ".join(sorted({path.suffix.lower() for path in saved_paths}))
    raise ValueError(
        "Formato no soportado. Suba un CSV de 12 derivaciones o el par .hea + .mat del mismo registro. "
        f"Extensiones recibidas: {suffixes or 'ninguna'}."
    )


# ---------------------------------------------------------------------------
# Etiquetas reales / comparación multilabel
# ---------------------------------------------------------------------------
def parse_reference_label_text(value: str | Iterable[str] | None) -> list[str]:
    """Convierte texto manual con clases o SNOMED a nombres del esquema 12-clases."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,;\n]+", value)
    else:
        parts = list(value)

    upper_to_name = {name.upper(): name for name in load.CLASS_NAMES}
    display_to_name = {group["display_name"].upper(): group["name"] for group in load.CLASS_GROUPS_12}
    found: list[str] = []
    seen = set()
    for part in parts:
        raw = str(part).strip()
        if not raw:
            continue
        code = load.normalize_snomed_code(raw)
        name = None
        if code in load.CODE_TO_CLASS_INDEX:
            name = load.CLASS_NAMES[load.CODE_TO_CLASS_INDEX[code]]
        else:
            key = raw.upper().replace(" ", "_")
            name = upper_to_name.get(key) or display_to_name.get(raw.upper())
        if name and name not in seen:
            found.append(name)
            seen.add(name)
    return found


def _class_order(names: Iterable[str]) -> list[str]:
    order = {name: i for i, name in enumerate(load.CLASS_NAMES)}
    return sorted(set(names), key=lambda n: order.get(n, 10_000))


def build_label_comparison(
    true_classes: Iterable[str] | None,
    predicted_rows: list[dict],
    source: str | None,
    dx_codes: Iterable[str] | None = None,
    manual_raw: str | None = None,
) -> dict:
    """Compara predicciones finales positivas vs etiquetas reales dentro del esquema."""
    predicted = _class_order(row["class"] for row in predicted_rows if int(row.get("prediction", 0)) == 1)
    true = _class_order(true_classes or [])
    has_truth = source is not None
    if not has_truth:
        return {
            "available": False,
            "source": None,
            "reason": "No hay etiquetas reales disponibles. En .hea se leen desde el campo Dx; para CSV puede escribir clases o SNOMED manualmente.",
            "true_classes": [],
            "predicted_classes": predicted,
        }

    true_set = set(true)
    pred_set = set(predicted)
    tp = _class_order(true_set & pred_set)
    fp = _class_order(pred_set - true_set)
    fn = _class_order(true_set - pred_set)
    union = true_set | pred_set

    if not true_set and not pred_set:
        precision = recall = f1 = jaccard = 1.0
    elif not pred_set:
        precision = 0.0
        recall = 0.0 if true_set else 1.0
        f1 = 0.0
        jaccard = 0.0 if union else 1.0
    elif not true_set:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        jaccard = 0.0
    else:
        precision = len(tp) / len(pred_set)
        recall = len(tp) / len(true_set)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        jaccard = len(tp) / len(union) if union else 1.0

    return {
        "available": True,
        "source": source,
        "manual_raw": manual_raw or "",
        "dx_codes": list(dx_codes or []),
        "true_classes": true,
        "predicted_classes": predicted,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "exact_match": pred_set == true_set,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
    }


# ---------------------------------------------------------------------------
# Gráficos / resultados
# ---------------------------------------------------------------------------
def _robust_ylim(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo, hi = -2.5, 2.5
    pad = max(0.5, 0.12 * float(hi - lo))
    lo = float(np.floor((lo - pad) * 2) / 2)
    hi = float(np.ceil((hi + pad) * 2) / 2)
    return lo, hi


def plot_ecg_array(signal: np.ndarray, output_dir: str | Path, stem: str) -> Path:
    """Grafica las 12 derivaciones con estilo de papel milimetrado ECG."""
    from matplotlib.ticker import MultipleLocator

    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != (load.WINDOW_LENGTH, load.NUM_LEADS):
        raise ValueError(f"Shape inválida para gráfico ECG: {signal.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(signal.shape[0]) / load.TARGET_FS

    fig, axes = plt.subplots(6, 2, figsize=(16, 11.5), sharex=True)
    fig.patch.set_facecolor("#fffafa")
    fig.suptitle(
        f"ECG de 12 derivaciones — {stem} · papel milimetrado",
        fontsize=15,
        fontweight="bold",
        color="#7f1d1d",
    )

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#fffafa")
        ax.plot(t, signal[:, i], linewidth=0.75, color="#111827", zorder=3)
        ax.set_title(LEADS[i], fontsize=10, loc="left", color="#7f1d1d", fontweight="bold")
        ax.set_xlim(0, load.WINDOW_SECONDS)
        y0, y1 = _robust_ylim(signal[:, i])
        ax.set_ylim(y0, y1)
        ax.xaxis.set_major_locator(MultipleLocator(0.20))
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        ax.yaxis.set_major_locator(MultipleLocator(0.50))
        ax.yaxis.set_minor_locator(MultipleLocator(0.10))
        ax.grid(which="minor", color="#f7c8c8", linewidth=0.35)
        ax.grid(which="major", color="#e48b8b", linewidth=0.75)
        ax.tick_params(axis="both", which="major", labelsize=7, colors="#7f1d1d", length=0)
        ax.tick_params(axis="both", which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_color("#e48b8b")
            spine.set_linewidth(0.6)
        ax.set_ylabel("z", fontsize=8, color="#7f1d1d")
    axes[-1, 0].set_xlabel("Tiempo (s)", fontsize=9, color="#7f1d1d")
    axes[-1, 1].set_xlabel("Tiempo (s)", fontsize=9, color="#7f1d1d")
    fig.text(
        0.5,
        0.008,
        "Cuadrícula ECG: 0.04 s por cuadro pequeño y 0.20 s por cuadro grande en el eje temporal. Amplitud mostrada normalizada por derivación.",
        ha="center",
        fontsize=8,
        color="#7f1d1d",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_path = output_dir / f"{secure_filename(stem)}_ecg_papel_milimetrado.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def threshold_values_for_class_names(
    class_names: list[str],
    threshold: float = 0.5,
    thresholds: dict | list | np.ndarray | None = None,
) -> np.ndarray:
    """Devuelve umbral por clase; si no hay thresholds calibrados usa fallback global."""
    values = np.full(len(class_names), float(threshold), dtype=np.float32)
    if thresholds is None:
        return values
    if isinstance(thresholds, dict):
        for i, name in enumerate(class_names):
            if name in thresholds:
                values[i] = float(thresholds[name])
            elif str(i) in thresholds:
                values[i] = float(thresholds[str(i)])
        return values
    arr = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if arr.size != len(class_names):
        raise ValueError(f"thresholds tiene {arr.size} valores; se esperaban {len(class_names)}")
    return arr


def _prediction_rows(probabilities: np.ndarray, class_names: list[str], thresholds: np.ndarray) -> list[dict]:
    rows = []
    for i, prob in enumerate(probabilities):
        group = load.CLASS_GROUPS_12[i]
        thr = float(thresholds[i])
        prob_f = float(prob)
        rows.append({
            "index": i,
            "class": class_names[i],
            "display_name": group["display_name"],
            "description": group["description"],
            "snomed_codes": group["codes"],
            "probability": prob_f,
            "probability_pct": round(prob_f * 100, 2),
            "threshold": thr,
            "threshold_pct": round(thr * 100, 2),
            "margin": float(prob_f - thr),
            "near_threshold": bool(abs(prob_f - thr) < 0.05),
            "prediction": int(prob_f >= thr),
        })
    return rows


def apply_normal_fallback(
    rows: list[dict],
    min_nsr_probability: float = 0.40,
) -> dict:
    """Postprocesamiento opcional para evitar salida multilabel vacía.

    En inferencia multilabel puede ocurrir que ninguna clase supere su umbral
    calibrado. Para la demo clínica/académica es más claro declarar una salida
    final cuando la clase normal tiene probabilidad razonable y todas las clases
    patológicas quedaron bajo su propio umbral. La regla NO modifica las
    probabilidades del modelo; solo añade NSR como decisión final postprocesada.
    """
    positives = [row for row in rows if int(row.get("prediction", 0)) == 1]
    info = {
        "enabled": True,
        "applied": False,
        "rule": "if_no_positive_and_NSR_probability_ge_min",
        "min_nsr_probability": float(min_nsr_probability),
        "added_class": None,
        "reason": "",
    }
    if positives:
        info["reason"] = "already_has_positive_predictions"
        return info
    nsr_row = next((row for row in rows if row.get("class") == "NSR"), None)
    if nsr_row is None:
        info["reason"] = "NSR_not_present_in_class_names"
        return info
    nsr_prob = float(nsr_row.get("probability", 0.0))
    if nsr_prob >= float(min_nsr_probability):
        nsr_row["prediction"] = 1
        nsr_row["postprocessed"] = True
        nsr_row["postprocess_reason"] = "normal_fallback_no_positive"
        info.update({
            "applied": True,
            "added_class": "NSR",
            "reason": "no_class_exceeded_threshold_and_NSR_probability_is_reasonable",
        })
    else:
        info["reason"] = "NSR_probability_below_min"
    return info


def run_prediction_from_saved_paths(
    saved_paths: list[Path],
    model_path: str | None,
    saved_dir: str | None,
    result_dir: str | Path,
    threshold: float = 0.5,
    reference_labels: str | Iterable[str] | None = None,
    thresholds: dict | list | np.ndarray | None = None,
    threshold_source: str | None = None,
    normal_fallback_min_prob: float | None = 0.40,
) -> dict:
    """Predice a partir de archivos ya guardados en disco."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_saved_input(saved_paths, result_dir)
    resolved_model = resolve_model_path(model_path=model_path, saved_dir=saved_dir)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, class_names = load_model(resolved_model, device=device)
    probabilities = predict_array(model, prepared["processed_signal"], device=device)
    threshold_values = threshold_values_for_class_names(class_names, threshold=threshold, thresholds=thresholds)
    rows = _prediction_rows(probabilities, class_names, threshold_values)
    raw_sorted_rows = sorted(rows, key=lambda item: item["probability"], reverse=True)
    raw_positives = [dict(row) for row in raw_sorted_rows if row["prediction"] == 1]
    normal_fallback = {"enabled": False, "applied": False}
    if normal_fallback_min_prob is not None and float(normal_fallback_min_prob) > 0:
        normal_fallback = apply_normal_fallback(rows, min_nsr_probability=float(normal_fallback_min_prob))
    sorted_rows = sorted(rows, key=lambda item: item["probability"], reverse=True)
    positives = [row for row in sorted_rows if row["prediction"] == 1]
    using_class_thresholds = thresholds is not None

    manual_raw = clean_path_text(reference_labels) if isinstance(reference_labels, (str, os.PathLike)) else ""
    manual_true = parse_reference_label_text(reference_labels)
    if manual_raw:
        label_comparison = build_label_comparison(manual_true, sorted_rows, "manual", manual_raw=manual_raw)
    elif prepared.get("true_label_source") == "wfdb_header_dx":
        label_comparison = build_label_comparison(
            prepared.get("matched_classes", []),
            sorted_rows,
            "wfdb_header_dx",
            dx_codes=prepared.get("dx_codes", []),
        )
    else:
        label_comparison = build_label_comparison(None, sorted_rows, None)

    plot_path = plot_ecg_array(
        prepared["processed_signal"],
        result_dir,
        prepared["record_name"],
    )

    age_value = prepared.get("age")
    try:
        age_clean = None if age_value is None or not np.isfinite(float(age_value)) else str(age_value)
    except Exception:
        age_clean = None if age_value is None else str(age_value)

    return {
        "status": "ok",
        "record_name": prepared["record_name"],
        "input_type": prepared["input_type"],
        "source_files": prepared["source_files"],
        "model_path": str(resolved_model),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "model_type": "CNN convencional equivalente" if checkpoint.get("config", {}).get("is_regular_conv") else "ResNet-34 1D tipo Hannun",
        "threshold": float(threshold),
        "thresholds": threshold_values.tolist(),
        "threshold_source": threshold_source or "fallback_global_0.5",
        "using_class_thresholds": bool(using_class_thresholds),
        "num_classes": load.NUM_CLASSES,
        "num_leads": load.NUM_LEADS,
        "target_sampling_rate": load.TARGET_FS,
        "input_length": load.WINDOW_LENGTH,
        "window_seconds": load.WINDOW_SECONDS,
        "class_names": class_names,
        "predictions": sorted_rows,
        "positive_predictions": positives,
        "raw_positive_predictions": raw_positives,
        "normal_fallback": normal_fallback,
        "top_predictions": raw_sorted_rows[:3],
        "plot_path": str(plot_path),
        "plot_url": None,
        "plot_style": "ecg_paper_grid",
        "converted_csv_path": str(prepared["converted_csv_path"]) if prepared.get("converted_csv_path") else None,
        "converted_csv_url": None,
        "original_sampling_rate": prepared.get("original_sampling_rate"),
        "lead_names": prepared.get("lead_names"),
        "processed_shape": list(prepared["processed_signal"].shape),
        "dx_codes": prepared.get("dx_codes", []),
        "matched_classes": prepared.get("matched_classes", []),
        "label_comparison": label_comparison,
        "patient_age": age_clean,
        "patient_sex": prepared.get("sex", "Unknown"),
        "technical_details": {
            "preprocessing": "12 derivaciones -> reordenamiento estándar -> 500 Hz -> z-score por derivación -> 5000 muestras",
            "label_schema": "cinc2020_12_grouped_snomed",
            "activation": "sigmoid por clase",
            "loss": "BCEWithLogitsLoss durante entrenamiento",
            "decision_rule": (
                "probabilidad >= threshold calibrado por clase"
                if using_class_thresholds else f"probabilidad >= {threshold}"
            ),
            "threshold_source": threshold_source or "fallback_global_0.5",
            "device": str(device),
            "plot_style": "papel milimetrado ECG",
        },
        "warning": "Uso académico. Estas predicciones no constituyen diagnóstico médico.",
    }


def run_prediction_from_uploads(
    file_storages: Iterable,
    model_path: str | None,
    saved_dir: str | None,
    upload_root: str | Path,
    results_root: str | Path,
    threshold: float = 0.5,
    reference_labels: str | Iterable[str] | None = None,
    thresholds: dict | list | np.ndarray | None = None,
    threshold_source: str | None = None,
    normal_fallback_min_prob: float | None = 0.40,
) -> dict:
    """Guarda archivos subidos, convierte si aplica y ejecuta inferencia."""
    job_id = uuid.uuid4().hex[:12]
    upload_dir = Path(upload_root) / job_id
    result_dir = Path(results_root) / job_id
    saved_paths = save_uploaded_files(file_storages, upload_dir)
    result = run_prediction_from_saved_paths(
        saved_paths=saved_paths,
        model_path=model_path,
        saved_dir=saved_dir,
        result_dir=result_dir,
        threshold=threshold,
        reference_labels=reference_labels,
        thresholds=thresholds,
        threshold_source=threshold_source,
        normal_fallback_min_prob=normal_fallback_min_prob,
    )
    result["job_id"] = job_id
    return result
