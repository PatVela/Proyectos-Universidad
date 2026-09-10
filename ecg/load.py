"""Carga, lectores y preprocesamiento para ECG PhysioNet/CinC 2020.

Este módulo actúa como fuente de verdad de datos para la estructura canónica
del proyecto::

    ecg/load.py

Responsabilidades:
    - definición del esquema SNOMED agrupado a 12 clases;
    - lectura de headers WFDB ``.hea`` y señales ``.mat``;
    - reordenamiento de 12 derivaciones;
    - remuestreo a 500 Hz;
    - normalización z-score por derivación;
    - padding/recorte a 5000 muestras;
    - construcción de HDF5 train/val/test con split multilabel estratificado.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "dataset2020"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cinc2020_12"

TARGET_FS = 500
WINDOW_SECONDS = 10
WINDOW_LENGTH = TARGET_FS * WINDOW_SECONDS
STANDARD_LEAD_ORDER = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]
NUM_LEADS = len(STANDARD_LEAD_ORDER)


# ---------------------------------------------------------------------------
# Esquema de etiquetas: 12 clases agrupadas por SNOMED-CT.
# ---------------------------------------------------------------------------
CLASS_GROUPS_12 = [
    {
        "name": "NSR",
        "display_name": "Normal sinus rhythm",
        "description": "Ritmo sinusal normal.",
        "codes": ["426783006"],
    },
    {
        "name": "LAD",
        "display_name": "Left axis deviation",
        "description": "Desviación del eje a la izquierda.",
        "codes": ["39732003"],
    },
    {
        "name": "MI",
        "display_name": "Myocardial infarction",
        "description": "Infarto de miocardio.",
        "codes": ["164865005"],
    },
    {
        "name": "TAb",
        "display_name": "T-wave abnormality",
        "description": "Anormalidad de onda T.",
        "codes": ["164934002"],
    },
    {
        "name": "AF",
        "display_name": "Atrial fibrillation / flutter group",
        "description": "Fibrilación/flutter auricular y variantes relacionadas.",
        "codes": [
            "164889003", "164890007", "195080001", "282825002",
            "426749004", "314208002",
        ],
    },
    {
        "name": "LVH",
        "display_name": "Left ventricular hypertrophy",
        "description": "Hipertrofia ventricular izquierda.",
        "codes": ["164873001"],
    },
    {
        "name": "VEctopy",
        "display_name": "Ventricular ectopy group",
        "description": "Extrasístoles/ectopia ventricular.",
        "codes": [
            "427172004", "17338001", "164884008", "11157007",
            "251180001", "251182009", "75532003", "81898007",
        ],
    },
    {
        "name": "AVBlock",
        "display_name": "Atrioventricular block group",
        "description": "Bloqueos auriculoventriculares.",
        "codes": [
            "270492004", "195042002", "233917008", "27885002",
            "54016002", "204384007",
        ],
    },
    {
        "name": "STach",
        "display_name": "Sinus tachycardia",
        "description": "Taquicardia sinusal.",
        "codes": ["427084000"],
    },
    {
        "name": "RBBB",
        "display_name": "Right bundle branch block group",
        "description": "Bloqueo de rama derecha; fusiona RBBB y CRBBB.",
        "codes": ["59118001", "713427006"],
    },
    {
        "name": "SB",
        "display_name": "Sinus bradycardia",
        "description": "Bradicardia sinusal.",
        "codes": ["426177001"],
    },
    {
        "name": "AEctopy_Junctional",
        "display_name": "Atrial ectopy / junctional rhythm group",
        "description": "Ectopia auricular, supraventricular y ritmos de unión.",
        "codes": [
            "284470004", "63593006", "713422000", "426664006",
            "29320008", "426995002", "251164006", "426648003",
            "195101003", "251268003", "251170000", "251168009",
            "251173003",
        ],
    },
]

CLASS_NAMES = [group["name"] for group in CLASS_GROUPS_12]
CLASS_DISPLAY_NAMES = [group["display_name"] for group in CLASS_GROUPS_12]
CLASS_DESCRIPTIONS = [group["description"] for group in CLASS_GROUPS_12]
NUM_CLASSES = len(CLASS_GROUPS_12)

CODE_TO_CLASS_INDEX: dict[str, int] = {}
for class_index, group in enumerate(CLASS_GROUPS_12):
    for code in group["codes"]:
        code = str(code).strip()
        if code in CODE_TO_CLASS_INDEX:
            raise ValueError(f"Código SNOMED duplicado en CLASS_GROUPS_12: {code}")
        CODE_TO_CLASS_INDEX[code] = class_index

CODE_TO_CLASS_NAME = {
    code: CLASS_NAMES[index]
    for code, index in CODE_TO_CLASS_INDEX.items()
}

EXCLUDED_GROUPS = [
    {
        "name": "TSV",
        "reason": "No alcanza 1000 ECGs en CINC2020 incluso agrupando códigos clínicamente relacionados.",
    },
    {
        "name": "TV",
        "reason": "No alcanza 1000 ECGs en CINC2020 incluso agrupando códigos clínicamente relacionados.",
    },
    {
        "name": "WPW",
        "reason": "No alcanza 1000 ECGs en CINC2020 incluso agrupando códigos clínicamente relacionados.",
    },
    {
        "name": "Noise",
        "reason": "Ruido no tiene código SNOMED-CT diagnóstico equivalente en el Challenge 2020.",
    },
]


# ---------------------------------------------------------------------------
# Etiquetas SNOMED -> vector multilabel.
# ---------------------------------------------------------------------------
def normalize_snomed_code(code) -> str | None:
    if code is None:
        return None
    text = str(code).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = text.replace(" ", "")
    return text if text else None


def parse_dx_field(raw_dx: str | Iterable[str] | None) -> list[str]:
    if raw_dx is None:
        return []
    parts = raw_dx.split(",") if isinstance(raw_dx, str) else list(raw_dx)
    codes, seen = [], set()
    for part in parts:
        code = normalize_snomed_code(part)
        if code is None or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def codes_to_vector(dx_codes: str | Iterable[str] | None) -> np.ndarray:
    """Mapea códigos SNOMED a vector multilabel binario de 12 clases."""
    y = np.zeros(NUM_CLASSES, dtype=np.float32)
    for code in parse_dx_field(dx_codes):
        index = CODE_TO_CLASS_INDEX.get(code)
        if index is not None:
            y[index] = 1.0
    return y


def matched_class_names(dx_codes: str | Iterable[str] | None) -> list[str]:
    vector = codes_to_vector(dx_codes)
    return [CLASS_NAMES[i] for i, active in enumerate(vector) if active > 0]


def class_codes_as_strings() -> list[str]:
    return [",".join(group["codes"]) for group in CLASS_GROUPS_12]


def class_groups_as_records() -> list[dict]:
    return [
        {
            "index": index,
            "class": group["name"],
            "display_name": group["display_name"],
            "description": group["description"],
            "snomed_codes": ",".join(group["codes"]),
        }
        for index, group in enumerate(CLASS_GROUPS_12)
    ]


# Backwards-friendly alias used by earlier drafts.
codes_to_12_vector = codes_to_vector


# ---------------------------------------------------------------------------
# Lectores ECG / preprocesamiento de señal.
# ---------------------------------------------------------------------------
def normalize_lead_name(name) -> str:
    raw = str(name).strip()
    mapping = {
        "I": "I", "II": "II", "III": "III",
        "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
        "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
    }
    return mapping.get(raw.upper(), raw)


def parse_header(hea_path: str | Path) -> dict:
    hea_path = Path(hea_path)
    with hea_path.open("r", encoding="utf-8-sig") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        raise ValueError("Header vacío")

    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Primera línea inválida: {lines[0]}")

    record_name = first[0]
    num_leads = int(first[1])
    sampling_freq = float(first[2])
    num_samples = int(first[3])

    if len(lines) < num_leads + 1:
        raise ValueError(f"Header incompleto: se esperaban {num_leads} derivaciones")

    lead_names = [normalize_lead_name(lines[i].split()[-1]) for i in range(1, num_leads + 1)]

    age = np.nan
    sex = "Unknown"
    dx_codes: list[str] = []
    for line in lines[num_leads + 1:]:
        if not line.startswith("#"):
            continue
        comment = line[1:].strip()
        lower = comment.lower()
        if lower.startswith("age:"):
            raw = comment.split(":", 1)[1].strip().replace(">", "").replace("<", "")
            try:
                value = float(raw)
                age = value if 0 <= value <= 120 else np.nan
            except ValueError:
                age = np.nan
        elif lower.startswith("sex:"):
            value = comment.split(":", 1)[1].strip()
            sex = value if value else "Unknown"
        elif lower.startswith("dx:"):
            dx_codes = parse_dx_field(comment.split(":", 1)[1].strip())

    return {
        "record_name": record_name,
        "num_leads": num_leads,
        "sampling_freq": sampling_freq,
        "num_samples": num_samples,
        "lead_names": lead_names,
        "age": age,
        "sex": sex,
        "dx_codes": dx_codes,
    }


def reorder_leads(signal: np.ndarray, lead_names: Iterable[str]) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"La señal debe ser 2D; shape={signal.shape}")
    normalized = [normalize_lead_name(name) for name in lead_names]
    missing = [lead for lead in STANDARD_LEAD_ORDER if lead not in normalized]
    if missing:
        raise ValueError(f"Faltan derivaciones estándar: {missing}; encontradas={normalized}")
    indices = [normalized.index(lead) for lead in STANDARD_LEAD_ORDER]
    return signal[:, indices]


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: int = TARGET_FS) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"La señal debe ser 2D; shape={signal.shape}")
    orig_fs = float(orig_fs)
    target_fs = float(target_fs)
    if orig_fs <= 0:
        raise ValueError(f"Frecuencia inválida: {orig_fs}")
    if np.isclose(orig_fs, target_fs):
        return signal
    ratio = Fraction(target_fs / orig_fs).limit_denominator(10000)
    return resample_poly(signal, ratio.numerator, ratio.denominator, axis=0)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Z-score por derivación, robusto frente a derivaciones constantes."""
    signal = np.asarray(signal, dtype=np.float64)
    mean = np.nanmean(signal, axis=0, keepdims=True)
    std = np.nanstd(signal, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    signal = (signal - mean) / std
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal.astype(np.float32, copy=False)


def fix_length(signal: np.ndarray, target_len: int = WINDOW_LENGTH, mode: str = "center") -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"La señal debe ser 2D; shape={signal.shape}")
    n = signal.shape[0]
    if n == target_len:
        return signal
    if n > target_len:
        if mode == "center":
            start = (n - target_len) // 2
        elif mode == "random":
            start = np.random.randint(0, n - target_len + 1)
        else:
            raise ValueError(f"Modo de recorte inválido: {mode}")
        return signal[start:start + target_len, :]
    return np.pad(signal, ((0, target_len - n), (0, 0)), mode="constant")


def preprocess_ecg_array(
    signal: np.ndarray,
    sampling_rate: float = TARGET_FS,
    lead_names: Iterable[str] | None = None,
    target_fs: int = TARGET_FS,
    target_length: int = WINDOW_LENGTH,
    normalize: bool = True,
) -> np.ndarray:
    """Convierte una señal 12-derivaciones a ``(5000, 12)`` lista para el modelo."""
    data = np.asarray(signal, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if data.ndim != 2:
        raise ValueError(f"Se esperaba ECG 2D; shape={data.shape}")

    # Acepta tanto (samples, leads) como (leads, samples).
    if data.shape[0] == NUM_LEADS and data.shape[1] != NUM_LEADS:
        data = data.T

    if lead_names is not None:
        data = reorder_leads(data, lead_names)
    elif data.shape[1] != NUM_LEADS:
        raise ValueError(f"Se requieren {NUM_LEADS} derivaciones; shape={data.shape}")

    data = resample_signal(data, sampling_rate, target_fs)
    if normalize:
        data = normalize_signal(data)
    data = fix_length(data, target_length, mode="center")
    return np.asarray(data, dtype=np.float32)


def load_mat_signal(mat_path: str | Path, num_leads: int = NUM_LEADS) -> np.ndarray:
    mat = loadmat(mat_path)
    if "val" not in mat:
        raise KeyError("No existe variable 'val' en el .mat")
    raw = np.asarray(mat["val"], dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Shape inesperado en .mat: {raw.shape}")
    if raw.shape[0] == num_leads:
        return raw.T
    if raw.shape[1] == num_leads:
        return raw
    # Fallback para el formato habitual WFDB challenge: (leads, samples).
    return raw.T


def load_wfdb_record(mat_path: str | Path, hea_path: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """Carga un registro WFDB challenge desde par ``.mat``/``.hea``."""
    mat_path = Path(mat_path)
    hea_path = Path(hea_path) if hea_path is not None else mat_path.with_suffix(".hea")
    meta = parse_header(hea_path)
    raw = load_mat_signal(mat_path, meta["num_leads"])
    processed = preprocess_ecg_array(
        raw,
        sampling_rate=meta["sampling_freq"],
        lead_names=meta["lead_names"],
    )
    return processed, meta


def read_csv_ecg(csv_path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    """Lee CSV para inferencia. Primera línea opcional: ``# Sampling Rate: 500 Hz``."""
    csv_path = Path(csv_path)
    sampling_rate = TARGET_FS
    with csv_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()

    skiprows = 0
    if first_line.startswith("#"):
        skiprows = 1
        lower = first_line.lower()
        if "sampling rate" in lower:
            try:
                sampling_rate = float(first_line.split(":", 1)[1].strip().split()[0])
            except Exception:
                sampling_rate = TARGET_FS

    df = pd.read_csv(csv_path, skiprows=skiprows)
    return df.values.astype(np.float32), sampling_rate, list(df.columns)


# ---------------------------------------------------------------------------
# Construcción de dataset HDF5.
# ---------------------------------------------------------------------------
def infer_source_db(hea_path: str | Path, data_dir: str | Path) -> str:
    parts = Path(os.path.relpath(hea_path, data_dir)).parts
    lower = [part.lower() for part in parts]
    if "training" in lower:
        idx = lower.index("training")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[0] if len(parts) > 1 else "Unknown"


def scan_records(data_dir: str | Path, drop_no_selected_labels: bool = False) -> tuple[list[dict], dict]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No existe data_dir: {data_dir}")

    hea_files = sorted(Path(p) for p in glob.glob(str(data_dir / "**" / "*.hea"), recursive=True))
    counters = {
        "headers_found": len(hea_files),
        "missing_mat": 0,
        "bad_header": 0,
        "bad_leads": 0,
        "no_selected_labels": 0,
        "kept_records": 0,
    }

    from tqdm import tqdm
    records: list[dict] = []
    for hea_path in tqdm(hea_files, desc="Escaneando headers", unit="reg"):
        mat_path = hea_path.with_suffix(".mat")
        if not mat_path.exists():
            counters["missing_mat"] += 1
            continue
        try:
            meta = parse_header(hea_path)
        except Exception as exc:
            counters["bad_header"] += 1
            tqdm.write(f"AVISO header inválido {hea_path}: {exc}")
            continue

        normalized_leads = [normalize_lead_name(lead) for lead in meta["lead_names"]]
        if meta["num_leads"] != NUM_LEADS or any(lead not in normalized_leads for lead in STANDARD_LEAD_ORDER):
            counters["bad_leads"] += 1
            continue

        label = codes_to_vector(meta["dx_codes"])
        matched = matched_class_names(meta["dx_codes"])
        if label.sum() == 0:
            counters["no_selected_labels"] += 1
            if drop_no_selected_labels:
                continue

        records.append({
            "hea_path": str(hea_path),
            "mat_path": str(mat_path),
            "record_name": meta["record_name"],
            "lead_names": meta["lead_names"],
            "sampling_freq": meta["sampling_freq"],
            "num_samples": meta["num_samples"],
            "age": meta["age"],
            "sex": meta["sex"],
            "source_db": infer_source_db(hea_path, data_dir),
            "dx_codes": meta["dx_codes"],
            "matched_classes": matched,
            "label": label,
        })

    counters["kept_records"] = len(records)
    return records, counters


def label_distribution(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    labels = np.stack([record["label"] for record in records])
    positives = labels.sum(axis=0).astype(int)
    total = labels.shape[0]
    return pd.DataFrame({
        "index": np.arange(NUM_CLASSES),
        "class": CLASS_NAMES,
        "display_name": CLASS_DISPLAY_NAMES,
        "positives": positives,
        "negatives": total - positives,
        "prevalence": positives / max(total, 1),
        "snomed_codes": class_codes_as_strings(),
    })


def source_distribution(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = [
        {
            "source_db": record["source_db"],
            "num_labels": int(record["label"].sum()),
            "all_zero": bool(record["label"].sum() == 0),
        }
        for record in records
    ]
    return (
        pd.DataFrame(rows)
        .groupby("source_db", dropna=False)
        .agg(records=("source_db", "size"), all_zero=("all_zero", "sum"), avg_labels=("num_labels", "mean"))
        .reset_index()
    )


def stratified_multilabel_split(
    labels: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Split 70/15/15 multilabel. Requiere iterative-stratification."""
    labels = np.asarray(labels, dtype=np.int32)
    if labels.shape[0] < 3:
        raise ValueError("No hay suficientes registros para train/val/test")
    if val_frac <= 0 or test_frac <= 0 or val_frac + test_frac >= 1:
        raise ValueError("Fracciones inválidas")

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:
        raise RuntimeError(
            "Falta iterative-stratification. Instala requirements.txt. "
            "No se usa fallback por fuente/subconjunto para evitar sesgo."
        ) from exc

    n = labels.shape[0]
    split_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    trainval_idx, test_idx = next(split_test.split(np.zeros((n, 1)), labels))

    val_frac_local = val_frac / (1.0 - test_frac)
    split_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac_local, random_state=random_state)
    train_sub, val_sub = next(split_val.split(np.zeros((len(trainval_idx), 1)), labels[trainval_idx]))

    return {
        "train": np.asarray(trainval_idx[train_sub], dtype=int),
        "val": np.asarray(trainval_idx[val_sub], dtype=int),
        "test": np.asarray(test_idx, dtype=int),
    }


def split_summary(records: list[dict], splits: dict[str, np.ndarray]) -> pd.DataFrame:
    labels = np.stack([record["label"] for record in records])
    rows = []
    for split_name, idx in splits.items():
        split_labels = labels[idx]
        positives = split_labels.sum(axis=0).astype(int)
        for i, name in enumerate(CLASS_NAMES):
            rows.append({
                "split": split_name,
                "records": int(len(idx)),
                "class_index": i,
                "class": name,
                "positives": int(positives[i]),
                "prevalence": float(positives[i] / max(len(idx), 1)),
            })
        rows.append({
            "split": split_name,
            "records": int(len(idx)),
            "class_index": -1,
            "class": "ALL_ZERO",
            "positives": int((split_labels.sum(axis=1) == 0).sum()),
            "prevalence": float((split_labels.sum(axis=1) == 0).mean()),
        })
    return pd.DataFrame(rows)


def split_assignments(records: list[dict], splits: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for split_name, indices in splits.items():
        for idx in indices:
            record = records[int(idx)]
            rows.append({
                "split": split_name,
                "index": int(idx),
                "record_name": record["record_name"],
                "source_db": record["source_db"],
                "dx_codes": ",".join(record["dx_codes"]),
                "matched_classes": ",".join(record["matched_classes"]),
                **{CLASS_NAMES[i]: int(record["label"][i]) for i in range(NUM_CLASSES)},
            })
    return pd.DataFrame(rows).sort_values(["split", "record_name"])


def _process_single_record(record: dict) -> dict:
    try:
        raw = load_mat_signal(record["mat_path"], record.get("num_leads", NUM_LEADS))
        signal = preprocess_ecg_array(
            raw,
            sampling_rate=record["sampling_freq"],
            lead_names=record["lead_names"],
            target_fs=TARGET_FS,
            target_length=WINDOW_LENGTH,
            normalize=True,
        )
        return {
            "ok": True,
            "signal": signal,
            "label": record["label"],
            "age": record["age"],
            "sex": record["sex"],
            "source_db": record["source_db"],
            "record_name": record["record_name"],
            "dx_codes": ",".join(record["dx_codes"]),
            "matched_classes": ",".join(record["matched_classes"]),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "record_name": record.get("record_name", "Unknown"),
            "error": str(exc),
        }


def _write_string_attr(h5_file, name: str, values: list[str]) -> None:
    import h5py
    dtype = h5py.string_dtype(encoding="utf-8")
    h5_file.attrs.create(name, np.asarray(values, dtype=dtype))


def _write_hdf5_attributes(h5_file, split_name: str, n_requested: int) -> None:
    h5_file.attrs["dataset"] = "PhysioNet/CinC Challenge 2020"
    h5_file.attrs["label_schema"] = "cinc2020_12_grouped_snomed"
    h5_file.attrs["label_mode"] = "multilabel"
    h5_file.attrs["problem_type"] = "multilabel_sigmoid_bce"
    h5_file.attrs["split"] = split_name
    h5_file.attrs["requested_records_before_signal_errors"] = int(n_requested)
    h5_file.attrs["sampling_rate"] = TARGET_FS
    h5_file.attrs["target_fs"] = TARGET_FS
    h5_file.attrs["window_seconds"] = WINDOW_SECONDS
    h5_file.attrs["window_length"] = WINDOW_LENGTH
    h5_file.attrs["num_leads"] = NUM_LEADS
    h5_file.attrs["num_classes"] = NUM_CLASSES
    h5_file.attrs["lead_order"] = ",".join(STANDARD_LEAD_ORDER)
    h5_file.attrs["normalization"] = "per-lead z-score before padding"
    h5_file.attrs["class_metadata_json"] = json.dumps(class_groups_as_records(), ensure_ascii=False)
    h5_file.attrs["excluded_groups_json"] = json.dumps(EXCLUDED_GROUPS, ensure_ascii=False)
    _write_string_attr(h5_file, "classes", CLASS_NAMES)
    _write_string_attr(h5_file, "class_display_names", CLASS_DISPLAY_NAMES)
    _write_string_attr(h5_file, "class_snomed_codes", class_codes_as_strings())


def process_split_to_hdf5(
    records: list[dict],
    indices: Iterable[int],
    output_path: str | Path,
    num_workers: int = 6,
    chunksize: int = 8,
    write_batch_size: int = 32,
) -> dict:
    import h5py
    from tqdm import tqdm

    indices = np.asarray(list(indices), dtype=int)
    selected = [records[int(i)] for i in indices]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as h5:
        chunk_n = min(16, max(1, len(selected)))
        signals_ds = h5.create_dataset(
            "signals",
            shape=(0, WINDOW_LENGTH, NUM_LEADS),
            maxshape=(None, WINDOW_LENGTH, NUM_LEADS),
            dtype="float32",
            chunks=(chunk_n, WINDOW_LENGTH, NUM_LEADS),
            compression="gzip",
            compression_opts=4,
        )
        labels_ds = h5.create_dataset(
            "labels",
            shape=(0, NUM_CLASSES),
            maxshape=(None, NUM_CLASSES),
            dtype="float32",
            chunks=(min(256, max(1, len(selected))), NUM_CLASSES),
        )
        ages_ds = h5.create_dataset("ages", shape=(0,), maxshape=(None,), dtype="float32")
        sexes_ds = h5.create_dataset("sexes", shape=(0,), maxshape=(None,), dtype=string_dtype)
        sources_ds = h5.create_dataset("source_dbs", shape=(0,), maxshape=(None,), dtype=string_dtype)
        names_ds = h5.create_dataset("record_names", shape=(0,), maxshape=(None,), dtype=string_dtype)
        dx_ds = h5.create_dataset("dx_codes", shape=(0,), maxshape=(None,), dtype=string_dtype)
        matched_ds = h5.create_dataset("matched_classes", shape=(0,), maxshape=(None,), dtype=string_dtype)

        _write_hdf5_attributes(h5, output_path.stem, len(selected))

        buffers = {
            "signals": [], "labels": [], "ages": [], "sexes": [],
            "source_dbs": [], "record_names": [], "dx_codes": [], "matched_classes": [],
        }
        n_ok, n_error = 0, 0
        error_examples = []

        def flush() -> None:
            nonlocal n_ok
            if not buffers["signals"]:
                return
            batch_size = len(buffers["signals"])
            old, new = n_ok, n_ok + batch_size
            for dataset in [signals_ds, labels_ds, ages_ds, sexes_ds, sources_ds, names_ds, dx_ds, matched_ds]:
                dataset.resize(new, axis=0)
            signals_ds[old:new] = np.stack(buffers["signals"]).astype(np.float32, copy=False)
            labels_ds[old:new] = np.stack(buffers["labels"]).astype(np.float32, copy=False)
            ages_ds[old:new] = np.asarray(buffers["ages"], dtype=np.float32)
            sexes_ds[old:new] = buffers["sexes"]
            sources_ds[old:new] = buffers["source_dbs"]
            names_ds[old:new] = buffers["record_names"]
            dx_ds[old:new] = buffers["dx_codes"]
            matched_ds[old:new] = buffers["matched_classes"]
            n_ok = new
            for values in buffers.values():
                values.clear()

        if num_workers == 1:
            iterator = map(_process_single_record, selected)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=num_workers)
            iterator = executor.map(_process_single_record, selected, chunksize=chunksize)

        try:
            for result in tqdm(iterator, total=len(selected), desc=f"Procesando {output_path.name}", unit="reg"):
                if not result["ok"]:
                    n_error += 1
                    if len(error_examples) < 50:
                        error_examples.append({"record_name": result["record_name"], "error": result["error"]})
                    tqdm.write(f"AVISO {result['record_name']}: {result['error']}")
                    continue
                buffers["signals"].append(result["signal"])
                buffers["labels"].append(result["label"])
                buffers["ages"].append(result["age"])
                buffers["sexes"].append(result["sex"])
                buffers["source_dbs"].append(result["source_db"])
                buffers["record_names"].append(result["record_name"])
                buffers["dx_codes"].append(result["dx_codes"])
                buffers["matched_classes"].append(result["matched_classes"])
                if len(buffers["signals"]) >= write_batch_size:
                    flush()
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        flush()
        h5.attrs["written_records"] = int(n_ok)
        h5.attrs["signal_processing_errors"] = int(n_error)

    print(f"Guardado {output_path}: {n_ok:,} OK, {n_error:,} errores")
    return {"path": str(output_path), "ok": n_ok, "errors": n_error, "error_examples": error_examples}


def write_reports(
    output_dir: str | Path,
    records: list[dict],
    counters: dict,
    splits: dict[str, np.ndarray] | None = None,
    drop_no_selected_labels: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(class_groups_as_records()).to_csv(output_dir / "class_mapping_12.csv", index=False)
    label_distribution(records).to_csv(output_dir / "label_distribution.csv", index=False)
    source_distribution(records).to_csv(output_dir / "source_distribution.csv", index=False)
    if splits is not None:
        split_summary(records, splits).to_csv(output_dir / "split_summary.csv", index=False)
        split_assignments(records, splits).to_csv(output_dir / "split_assignments.csv", index=False)

    summary = {
        "dataset": "PhysioNet/CinC Challenge 2020",
        "label_schema": "cinc2020_12_grouped_snomed",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "classes": class_groups_as_records(),
        "excluded_groups": EXCLUDED_GROUPS,
        "target_fs": TARGET_FS,
        "window_seconds": WINDOW_SECONDS,
        "window_length": WINDOW_LENGTH,
        "num_leads": NUM_LEADS,
        "lead_order": STANDARD_LEAD_ORDER,
        "normalization": "per-lead z-score before padding",
        "split_method": "MultilabelStratifiedShuffleSplit by label vector, not by source",
        "drop_no_selected_labels": bool(drop_no_selected_labels),
        "scan_counters": counters,
        "splits": {name: int(len(idx)) for name, idx in splits.items()} if splits is not None else None,
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_datasets(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    drop_no_selected_labels: bool = False,
    scan_only: bool = False,
    workers: int = 6,
    chunksize: int = 8,
    write_batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Función programática usada por ``examples/cinc2020/build_datasets.py``."""
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    print("\n" + "=" * 80)
    print("PREPROCESAMIENTO CINC2020 — 12 CLASES AGRUPADAS")
    print("=" * 80)
    print(f"Dataset     : {data_dir}")
    print(f"Salida      : {output_dir}")
    print(f"Señal       : {NUM_LEADS} leads, {TARGET_FS} Hz, {WINDOW_LENGTH} muestras")
    print(f"Clases      : {NUM_CLASSES} -> {CLASS_NAMES}")
    print(f"All-zero    : {'excluir' if drop_no_selected_labels else 'conservar'}")

    records, counters = scan_records(data_dir, drop_no_selected_labels=drop_no_selected_labels)
    print("\nEscaneo:")
    for key, value in counters.items():
        print(f"  {key:<24}: {value:,}")
    if not records:
        raise RuntimeError("No quedaron registros válidos tras el escaneo")

    dist = label_distribution(records)
    print("\nDistribución por clase:")
    print(dist[["index", "class", "positives", "prevalence"]].to_string(index=False))

    labels = np.stack([record["label"] for record in records])
    empty = [CLASS_NAMES[i] for i, count in enumerate(labels.sum(axis=0)) if count <= 0]
    if empty:
        raise RuntimeError("Clases sin positivos: " + ", ".join(empty))

    if scan_only:
        write_reports(output_dir, records, counters, splits=None, drop_no_selected_labels=drop_no_selected_labels)
        return {"records": len(records), "counters": counters, "splits": None}

    splits = stratified_multilabel_split(labels, val_frac=val_frac, test_frac=test_frac, random_state=random_state)
    print("\nSplits:")
    for split_name, idx in splits.items():
        print(f"  {split_name:<5}: {len(idx):,}")

    write_reports(output_dir, records, counters, splits=splits, drop_no_selected_labels=drop_no_selected_labels)

    processing = []
    for split_name in ["train", "val", "test"]:
        processing.append(process_split_to_hdf5(
            records=records,
            indices=splits[split_name],
            output_path=output_dir / f"{split_name}.h5",
            num_workers=workers,
            chunksize=chunksize,
            write_batch_size=write_batch_size,
        ))

    (output_dir / "signal_processing_summary.json").write_text(
        json.dumps(processing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"records": len(records), "counters": counters, "splits": {k: len(v) for k, v in splits.items()}, "processing": processing}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Construye HDF5 CINC2020-12 desde archivos .hea/.mat")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR), help="Raíz local con CINC2020 training/")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio de salida")
    parser.add_argument("--drop_no_selected_labels", action="store_true", help="Excluir ECGs sin ninguna de las 12 clases")
    parser.add_argument("--scan_only", action="store_true", help="Solo escanear headers y escribir reportes")
    parser.add_argument("--workers", type=int, default=6, help="Procesos de preprocesamiento")
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--write_batch_size", type=int, default=32)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args(argv)

    build_datasets(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        drop_no_selected_labels=args.drop_no_selected_labels,
        scan_only=args.scan_only,
        workers=args.workers,
        chunksize=args.chunksize,
        write_batch_size=args.write_batch_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    freeze_support()
    main()
