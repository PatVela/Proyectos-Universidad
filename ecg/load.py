"""Carga, lectores y preprocesamiento para ECG PhysioNet/CinC 2020.

Este módulo actúa como fuente de verdad de datos para la estructura canónica
del proyecto::

    ecg/load.py

Responsabilidades:
    - definición del esquema SNOMED agrupado a 12 clases (v2: cubre las 27
      clases puntuadas del Challenge 2020);
    - lectura de headers WFDB ``.hea`` y señales ``.mat`` (incluye ganancia
      ADC y baseline por derivación);
    - reordenamiento de 12 derivaciones;
    - remuestreo a 500 Hz;
    - conversión a unidades físicas (mV) + clip (por defecto) o z-score;
    - padding/recorte a 5000 muestras o ventanas deslizantes;
    - construcción de HDF5 train/val/test con split multilabel estratificado.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt


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
# Esquema de etiquetas: 12 clases agrupadas por SNOMED-CT (v2).
#
# La v1 mapeaba ~40 códigos y dejaba ~21% de registros CINC2020 como vectores
# all-zero (LBBB, TInv, IRBBB, Brady genérica, NSSTTA, STD/STE, LQT, OldMI,
# isquemias, etc. quedaban sin clase), además de ignorar 13 de las 27 clases
# puntuadas oficiales del Challenge 2020. Eso introduce ruido de etiquetas
# masivo: ECG claramente anormales etiquetados como "nada", y clases como TAb
# entrenadas con TAb=0 para la mayoría de las anomalías ST-T reales.
#
# La v2 agrupa códigos clínicamente relacionados para cubrir las 27 clases
# puntuadas oficiales
# (https://github.com/physionetchallenges/evaluation-2020/blob/master/dx_mapping_scored.csv)
# manteniendo 12 salidas. All-zero esperado: <1% (solo diagnósticos
# no-ECG como HF/HVD/CHD o WPW aislado).
# ---------------------------------------------------------------------------
LABEL_SCHEMA = "cinc2020_12_grouped_snomed_v2"
LEGACY_LABEL_SCHEMA = "cinc2020_12_grouped_snomed"

CLASS_GROUPS_12 = [
    {
        "name": "NSR",
        "display_name": "Sinus rhythm (normal / arrhythmia)",
        "description": "Ritmo sinusal normal o arritmia sinusal benigna.",
        "codes": ["426783006", "427393009"],
    },
    {
        "name": "AxisDev",
        "display_name": "Cardiac axis deviation group",
        "description": "Desviación del eje (LAD/RAD/indeterminado) y bloqueos fasciculares.",
        "codes": ["39732003", "445118002", "47665007", "445211001", "251200008"],
    },
    {
        "name": "MI",
        "display_name": "Myocardial infarction / ischemia group",
        "description": "Infarto (incluye antiguo/anterior/agudo), isquemia y Q anormal.",
        "codes": [
            "164865005", "57054005", "164867002", "54329005",
            "164861001", "413444003", "413844008", "426434006",
            "425419005", "425623009", "164917005",
        ],
    },
    {
        "name": "TAb",
        "display_name": "ST-T / repolarization abnormality group",
        "description": "Anomalías ST-T y de repolarización (TAb/TInv/NSSTTA/STD/STE/QT).",
        "codes": [
            "164934002", "59931005", "428750005", "429622005",
            "164931005", "164930006", "55930002", "704997005",
            "111975006", "77867006", "164937009", "251259000",
            "428417006",
        ],
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
        "display_name": "Ventricular hypertrophy / enlargement group",
        "description": "Hipertrofia/crecimiento ventricular y auricular, strain.",
        "codes": [
            "164873001", "370365005", "446813000", "67741000119109",
            "253352002", "195126007", "89792004", "266249003",
            "253339007", "446358003",
        ],
    },
    {
        "name": "VEctopy",
        "display_name": "Ventricular ectopy / tachycardia group",
        "description": "Ectopia ventricular y taquiarritmias ventriculares (TV/FV incluidas para no etiquetarlas como sanas).",
        "codes": [
            "427172004", "17338001", "164884008", "11157007",
            "251180001", "251182009", "75532003", "81898007",
            "164895002", "425856008", "164896001", "111288001",
            "49260003", "13640000",
        ],
    },
    {
        "name": "AVBlock",
        "display_name": "Atrioventricular / sinoatrial block group",
        "description": "Bloqueos AV (incluye PR prolongado) y sinoauriculares.",
        "codes": [
            "270492004", "195042002", "233917008", "27885002",
            "54016002", "204384007", "164947007", "65778007",
        ],
    },
    {
        "name": "STach",
        "display_name": "Sinus tachycardia",
        "description": "Taquicardia sinusal.",
        "codes": ["427084000"],
    },
    {
        "name": "BBB",
        "display_name": "Bundle branch block / QRS abnormality group",
        "description": "Bloqueos de rama (der/izq, completos/incompletos), conducción inespecífica y QRS anormal/bajo voltaje.",
        "codes": [
            "59118001", "713427006", "713426002", "164909002",
            "251120003", "6374002", "698252002", "82226007",
            "251146004", "164951009",
        ],
    },
    {
        "name": "SB",
        "display_name": "Bradycardia group",
        "description": "Bradicardia sinusal/genérica, disfunción sinusal y síndrome bradi-taqui.",
        "codes": ["426177001", "426627000", "60423000", "74615001"],
    },
    {
        "name": "AEctopy_Junctional",
        "display_name": "Atrial ectopy / junctional / SVT / pacing group",
        "description": "Ectopia auricular/supraventricular, ritmos de unión, TSV y ritmos de marcapasos.",
        "codes": [
            "284470004", "63593006", "713422000", "426664006",
            "29320008", "426995002", "251164006", "426648003",
            "195101003", "251268003", "251170000", "251168009",
            "251173003", "426761007", "67198005", "10370003",
            "251266004",
        ],
    },
]

# Esquema v1 conservado solo para auditoría/migración de checkpoints antiguos.
CLASS_GROUPS_12_V1_LEGACY = [
    {"name": "NSR", "codes": ["426783006"]},
    {"name": "LAD", "codes": ["39732003"]},
    {"name": "MI", "codes": ["164865005"]},
    {"name": "TAb", "codes": ["164934002"]},
    {"name": "AF", "codes": ["164889003", "164890007", "195080001", "282825002", "426749004", "314208002"]},
    {"name": "LVH", "codes": ["164873001"]},
    {"name": "VEctopy", "codes": ["427172004", "17338001", "164884008", "11157007", "251180001", "251182009", "75532003", "81898007"]},
    {"name": "AVBlock", "codes": ["270492004", "195042002", "233917008", "27885002", "54016002", "204384007"]},
    {"name": "STach", "codes": ["427084000"]},
    {"name": "RBBB", "codes": ["59118001", "713427006"]},
    {"name": "SB", "codes": ["426177001"]},
    {"name": "AEctopy_Junctional", "codes": ["284470004", "63593006", "713422000", "426664006", "29320008", "426995002", "251164006", "426648003", "195101003", "251268003", "251170000", "251168009", "251173003"]},
]
LEGACY_CLASS_NAMES = [g["name"] for g in CLASS_GROUPS_12_V1_LEGACY]

# Las 27 clases puntuadas oficiales del Challenge 2020 (dx_mapping_scored.csv).
# Cada entrada: (código SNOMED, abreviatura). Útil para mapear grupos -> códigos
# oficiales en examples/cinc2020/challenge_score.py.
SCORED_27_CODES = [
    ("270492004", "IAVB"), ("164889003", "AF"), ("164890007", "AFL"),
    ("426627000", "Brady"), ("713427006", "CRBBB"), ("713426002", "IRBBB"),
    ("445118002", "LAnFB"), ("39732003", "LAD"), ("164909002", "LBBB"),
    ("251146004", "LQRSV"), ("698252002", "NSIVCB"), ("10370003", "PR"),
    ("284470004", "PAC"), ("427172004", "PVC"), ("164947007", "LPR"),
    ("111975006", "LQT"), ("164917005", "QAb"), ("47665007", "RAD"),
    ("59118001", "RBBB"), ("427393009", "SA"), ("426177001", "SB"),
    ("426783006", "NSR"), ("427084000", "STach"), ("63593006", "SVPB"),
    ("164934002", "TAb"), ("59931005", "TInv"), ("17338001", "VPB"),
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
        "name": "WPW",
        "reason": "WPW/preexcitación (74390002, 195060002) es muy infrecuente en CINC2020 y morfológicamente singular; queda fuera del esquema v2.",
    },
    {
        "name": "NonECG_Diagnoses",
        "reason": "Diagnósticos clínicos no-ECG (p.ej. HF 84114007, HVD 368009, CHD 53741008, TIA 266257000) no describen morfología del trazado.",
    },
    {
        "name": "Noise",
        "reason": "Ruido no tiene código SNOMED-CT diagnóstico equivalente en el Challenge 2020.",
    },
]

# Normalización por defecto del esquema v2.
DEFAULT_NORM_MODE = "physical"
PHYSICAL_CLIP_MV = 5.0
DEFAULT_ADC_GAIN = 1000.0
BANDPASS_LO_HZ = 0.5
BANDPASS_HI_HZ = 50.0


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


def preprocessing_for_schema(label_schema: str | None) -> dict:
    """Devuelve parámetros de preprocesamiento compatibles con un checkpoint.

    - Esquema v2 (o desconocido/nuevo): ``physical`` + bandpass.
    - Esquema v1 legacy: ``per_lead_zscore`` sin bandpass.

    Usar SIEMPRE esta función en inferencia (webapp/predict) para no mezclar
    un checkpoint v1 con preprocesamiento v2 (o viceversa), lo que produce
    predicciones basura silenciosas.
    """
    if label_schema == LEGACY_LABEL_SCHEMA:
        return {"norm_mode": "per_lead_zscore", "bandpass": False}
    return {"norm_mode": DEFAULT_NORM_MODE, "bandpass": True}


def is_legacy_schema(label_schema: str | None) -> bool:
    return str(label_schema or "") == LEGACY_LABEL_SCHEMA


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

    adc_gains: list[float] = []
    adc_baselines: list[float] = []
    adc_units: list[str] = []
    for i in range(1, num_leads + 1):
        gain, baseline, units = parse_adc_spec(lines[i])
        adc_gains.append(gain)
        adc_baselines.append(baseline)
        adc_units.append(units)

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
        "adc_gains": adc_gains,
        "adc_baselines": adc_baselines,
        "adc_units": adc_units,
        "age": age,
        "sex": sex,
        "dx_codes": dx_codes,
    }


def parse_adc_spec(lead_line: str) -> tuple[float, float, str]:
    """Extrae (ganancia, baseline, unidades) de una línea de derivación WFDB.

    Ejemplo: ``A0001.mat 16+24 1000/mV 16 0 28 -1716 0 I`` -> (1000.0, 0.0, 'mV').
    Acepta variantes como ``1000.0(0)/mV``. Si algo falla, usa ganancia 1000 y
    baseline 0 (valores uniformes observados en los 6 subconjuntos CINC2020).
    """
    tokens = lead_line.split()
    gain = DEFAULT_ADC_GAIN
    baseline = 0.0
    units = "mV"
    try:
        if len(tokens) >= 3:
            match = re.match(r"([0-9.eE+-]+)(?:\\(([^)]*)\\))?(?:/(.*))?", tokens[2])
            if match:
                gain = float(match.group(1))
                if match.group(2) not in (None, ""):
                    try:
                        baseline = float(match.group(2))
                    except ValueError:
                        pass
                if match.group(3):
                    units = match.group(3)
        if len(tokens) >= 5:
            # Quinto token = adc_zero (baseline digital). Prevalece si es numérico.
            try:
                baseline = float(tokens[4])
            except ValueError:
                pass
    except (ValueError, IndexError):
        pass
    if not np.isfinite(gain) or gain == 0:
        gain = DEFAULT_ADC_GAIN
    if not np.isfinite(baseline):
        baseline = 0.0
    return float(gain), float(baseline), str(units)


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
    """Z-score por derivación, robusto frente a derivaciones constantes.

    Modo legacy (``per_lead_zscore``): elimina offsets por derivación pero
    también destruye amplitudes absolutas (criterios de voltaje de LVH/LQRSV)
    y relativas entre derivaciones (eje eléctrico). Se conserva solo para
    compatibilidad con checkpoints entrenados con el esquema v1.
    """
    signal = np.asarray(signal, dtype=np.float64)
    mean = np.nanmean(signal, axis=0, keepdims=True)
    std = np.nanstd(signal, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    signal = (signal - mean) / std
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal.astype(np.float32, copy=False)


def normalize_signal_global(signal: np.ndarray) -> np.ndarray:
    """Z-score global del registro (media/std sobre las 12 derivaciones).

    Preserva amplitudes *relativas* entre derivaciones (eje eléctrico),
    pero no absolutas. Intermedio entre ``per_lead_zscore`` y ``physical``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    mean = float(np.nanmean(signal))
    std = float(np.nanstd(signal))
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    signal = (signal - mean) / std
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal.astype(np.float32, copy=False)


def to_physical_units(
    signal: np.ndarray,
    adc_gains: Iterable[float] | None = None,
    adc_baselines: Iterable[float] | None = None,
) -> np.ndarray:
    """Convierte valores digitales WFDB a mV: ``(digital - baseline) / gain``.

    En los 6 subconjuntos CINC2020 la ganancia es uniformemente 1000/mV con
    baseline 0, pero se lee del header por robustez.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if adc_gains is None:
        gains = np.full(signal.shape[1], DEFAULT_ADC_GAIN)
    else:
        gains = np.asarray(list(adc_gains), dtype=np.float64)
    if adc_baselines is None:
        baselines = np.zeros(signal.shape[1])
    else:
        baselines = np.asarray(list(adc_baselines), dtype=np.float64)
    if gains.shape[0] != signal.shape[1] or baselines.shape[0] != signal.shape[1]:
        raise ValueError(
            f"Gains/baselines ({gains.shape[0]}) no coinciden con derivaciones ({signal.shape[1]})"
        )
    gains = np.where((~np.isfinite(gains)) | (gains == 0), DEFAULT_ADC_GAIN, gains)
    baselines = np.where(~np.isfinite(baselines), 0.0, baselines)
    physical = (signal - baselines[None, :]) / gains[None, :]
    return np.nan_to_num(physical, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float = TARGET_FS,
    lo_hz: float = BANDPASS_LO_HZ,
    hi_hz: float = BANDPASS_HI_HZ,
    order: int = 3,
) -> np.ndarray:
    """Pasa-banda Butterworth de fase cero (SOS + sosfiltfilt).

    Atenúa deriva de línea base (<0.5 Hz) y ruido muscular/red (>50 Hz).
    Requiere señal de al menos ~2 s; si es más corta, la devuelve intacta.
    """
    signal = np.asarray(signal, dtype=np.float64)
    nyquist = float(sampling_rate) / 2.0
    if signal.shape[0] < int(2 * sampling_rate):
        return signal.astype(np.float32, copy=False)
    lo = min(max(float(lo_hz) / nyquist, 1e-4), 0.99)
    hi = min(max(float(hi_hz) / nyquist, 1e-4), 0.99)
    if lo >= hi:
        raise ValueError(f"Banda inválida: lo={lo_hz} hi={hi_hz} fs={sampling_rate}")
    sos = butter(int(order), [lo, hi], btype="band", output="sos")
    filtered = sosfiltfilt(sos, signal, axis=0)
    return np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def compute_window_starts(
    n_samples: int,
    target_length: int = WINDOW_LENGTH,
    windows_max: int = 1,
) -> list[int]:
    """Inicios de ventana (en muestras) equiespaciados sobre la señal.

    - Señal corta (``n <= target``): un único inicio 0 (luego se aplica pad).
    - ``windows_max <= 1``: ventana centrada (comportamiento legacy).
    - Señal larga: hasta ``windows_max`` ventanas equiespaciadas que cubren
      de inicio a fin (incluye primera y última muestra).
    """
    n_samples = int(n_samples)
    target_length = int(target_length)
    if n_samples <= target_length or int(windows_max) <= 1:
        if n_samples <= target_length:
            return [0]
        return [(n_samples - target_length) // 2]
    windows_max = int(windows_max)
    span = n_samples - target_length
    if windows_max == 1 or span <= 0:
        return [span // 2]
    # Ventanas no solapadas si caben; si no, solapadas equiespaciadas.
    non_overlap = span // target_length + 1
    k = min(windows_max, max(1, non_overlap))
    if k == 1:
        return [span // 2]
    return [int(round(span * i / (k - 1))) for i in range(k)]


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
    target_length: int | None = WINDOW_LENGTH,
    normalize: bool = True,
    norm_mode: str = DEFAULT_NORM_MODE,
    adc_gains: Iterable[float] | None = None,
    adc_baselines: Iterable[float] | None = None,
    units: str = "digital",
    bandpass: bool = True,
    clip_mv: float = PHYSICAL_CLIP_MV,
    crop_mode: str = "center",
) -> np.ndarray:
    """Convierte una señal 12-derivaciones a lista para el modelo.

    Modos de normalización (``norm_mode``):

    - ``\"physical\"`` (defecto v2): convierte a mV con ganancia/baseline del
      header, aplica pasa-banda 0.5–50 Hz y clip ±``clip_mv``. Preserva
      amplitudes absolutas y relativas (voltaje y eje eléctrico).
    - ``\"global_zscore\"``: z-score global del registro (preserva eje).
    - ``\"per_lead_zscore\"``: legacy v1 (requiere ``normalize=True``);
      imprescindible para checkpoints entrenados con el esquema v1.

    Si ``target_length`` es ``None`` no se recorta/rellena (útil para extraer
    ventanas deslizantes sobre el registro completo).
    """
    data = np.asarray(signal, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if data.ndim != 2:
        raise ValueError(f"Se esperaba ECG 2D; shape={data.shape}")

    # Acepta tanto (samples, leads) como (leads, samples).
    if data.shape[0] == NUM_LEADS and data.shape[1] != NUM_LEADS:
        data = data.T

    if lead_names is not None:
        # Reordena gains/baselines junto con la señal si vienen en orden del header.
        names = [normalize_lead_name(n) for n in lead_names]
        if adc_gains is not None and adc_baselines is not None and len(names) == len(list(adc_gains)):
            order = [names.index(lead) for lead in STANDARD_LEAD_ORDER]
            adc_gains = [list(adc_gains)[k] for k in order]
            adc_baselines = [list(adc_baselines)[k] for k in order]
        data = reorder_leads(data, lead_names)
    elif data.shape[1] != NUM_LEADS:
        raise ValueError(f"Se requieren {NUM_LEADS} derivaciones; shape={data.shape}")

    norm_mode = str(norm_mode or DEFAULT_NORM_MODE).lower()
    if norm_mode not in {"physical", "global_zscore", "per_lead_zscore"}:
        raise ValueError(f"norm_mode inválido: {norm_mode}")

    if normalize and norm_mode == "physical" and str(units).lower() != "mv":
        data = to_physical_units(data, adc_gains=adc_gains, adc_baselines=adc_baselines)

    data = resample_signal(data, sampling_rate, target_fs)

    if normalize:
        if bandpass and norm_mode == "physical":
            data = bandpass_filter(data, sampling_rate=target_fs)
        if norm_mode == "physical":
            data = np.clip(np.asarray(data, dtype=np.float32), -float(clip_mv), float(clip_mv))
        elif norm_mode == "global_zscore":
            data = normalize_signal_global(data)
        else:
            data = normalize_signal(data)

    if target_length is not None:
        data = fix_length(data, int(target_length), mode=crop_mode)
    return np.asarray(data, dtype=np.float32)


def preprocess_to_windows(
    signal: np.ndarray,
    sampling_rate: float = TARGET_FS,
    lead_names: Iterable[str] | None = None,
    target_fs: int = TARGET_FS,
    target_length: int = WINDOW_LENGTH,
    windows_max: int = 1,
    **preprocess_kwargs,
) -> tuple[np.ndarray, list[int]]:
    """Preprocesa el registro completo y extrae hasta ``windows_max`` ventanas.

    Devuelve ``(ventanas, inicios)`` con ``ventanas`` de forma
    ``(K, target_length, 12)``. Con ``windows_max=1`` equivale al recorte
    centrado legacy.
    """
    full = preprocess_ecg_array(
        signal,
        sampling_rate=sampling_rate,
        lead_names=lead_names,
        target_fs=target_fs,
        target_length=None,
        **preprocess_kwargs,
    )
    starts = compute_window_starts(full.shape[0], target_length, windows_max)
    windows = []
    for start in starts:
        piece = full[start:start + target_length, :]
        if piece.shape[0] < target_length:
            piece = np.pad(piece, ((0, target_length - piece.shape[0]), (0, 0)), mode="constant")
        windows.append(piece.astype(np.float32, copy=False))
    return np.stack(windows).astype(np.float32, copy=False), [int(s) for s in starts]


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


def load_wfdb_record(
    mat_path: str | Path,
    hea_path: str | Path | None = None,
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
    target_length: int | None = WINDOW_LENGTH,
    crop_mode: str = "center",
) -> tuple[np.ndarray, dict]:
    """Carga un registro WFDB challenge desde par ``.mat``/``.hea``."""
    mat_path = Path(mat_path)
    hea_path = Path(hea_path) if hea_path is not None else mat_path.with_suffix(".hea")
    meta = parse_header(hea_path)
    raw = load_mat_signal(mat_path, meta["num_leads"])
    processed = preprocess_ecg_array(
        raw,
        sampling_rate=meta["sampling_freq"],
        lead_names=meta["lead_names"],
        target_length=target_length,
        norm_mode=norm_mode,
        adc_gains=meta.get("adc_gains"),
        adc_baselines=meta.get("adc_baselines"),
        units="digital",
        bandpass=bandpass,
        crop_mode=crop_mode,
    )
    meta["norm_mode"] = norm_mode
    meta["bandpass"] = bool(bandpass)
    return processed, meta


def load_wfdb_record_windows(
    mat_path: str | Path,
    hea_path: str | Path | None = None,
    windows_max: int = 4,
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
) -> tuple[np.ndarray, dict]:
    """Carga un registro WFDB y devuelve ``(ventanas, meta)``.

    ``ventanas`` tiene forma ``(K, 5000, 12)`` con K<=``windows_max``.
    Para registros de 10 s, K=1 (idéntico a :func:`load_wfdb_record`).
    """
    mat_path = Path(mat_path)
    hea_path = Path(hea_path) if hea_path is not None else mat_path.with_suffix(".hea")
    meta = parse_header(hea_path)
    raw = load_mat_signal(mat_path, meta["num_leads"])
    windows, starts = preprocess_to_windows(
        raw,
        sampling_rate=meta["sampling_freq"],
        lead_names=meta["lead_names"],
        windows_max=windows_max,
        norm_mode=norm_mode,
        adc_gains=meta.get("adc_gains"),
        adc_baselines=meta.get("adc_baselines"),
        units="digital",
        bandpass=bandpass,
    )
    meta["norm_mode"] = norm_mode
    meta["bandpass"] = bool(bandpass)
    meta["window_starts"] = starts
    meta["num_windows"] = int(windows.shape[0])
    return windows, meta


def read_csv_ecg(csv_path: str | Path) -> tuple[np.ndarray, float, list[str], dict]:
    """Lee CSV para inferencia.

    Líneas iniciales ``#`` (todas las que haya):

    - ``# Sampling Rate: 500 Hz`` -> frecuencia de muestreo;
    - ``# Preprocessed: <norm_mode>`` -> marcador escrito por la webapp al
      convertir WFDB->CSV; si está presente, la señal ya está en 500 Hz /
      5000 muestras / 12 derivaciones y NO debe reprocesarse.

    Devuelve ``(values, sampling_rate, lead_names, csv_meta)``.
    """
    csv_path = Path(csv_path)
    sampling_rate: float = TARGET_FS
    skiprows = 0
    comment_lines: list[str] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text.startswith("#"):
                break
            skiprows += 1
            comment_lines.append(text)
            lower = text.lower()
            if "sampling rate" in lower:
                try:
                    sampling_rate = float(text.split(":", 1)[1].strip().split()[0])
                except Exception:
                    sampling_rate = TARGET_FS

    preprocessed_mode: str | None = None
    for text in comment_lines:
        if text.lower().startswith("# preprocessed:"):
            preprocessed_mode = text.split(":", 1)[1].strip().lower() or "unknown"
            # Formato "physical, bandpass=true, units=mV" -> modo es primer token.
            preprocessed_mode = preprocessed_mode.split(",")[0].strip().split()[0]

    df = pd.read_csv(csv_path, skiprows=skiprows)
    csv_meta = {
        "sampling_rate": float(sampling_rate),
        "preprocessed_mode": preprocessed_mode,
        "comment_lines": comment_lines,
    }
    return df.values.astype(np.float32), float(sampling_rate), list(df.columns), csv_meta


def infer_csv_units(values: np.ndarray) -> str:
    """Heurística de unidades para CSV crudos de usuario.

    - Si la amplitud máxima supera 20 -> valores digitales/ADC (requieren
      conversión con ganancia 1000) -> ``\"digital\"``.
    - En otro caso se asumen milivoltios -> ``\"mv\"``.
    """
    try:
        peak = float(np.nanmax(np.abs(np.asarray(values, dtype=np.float64))))
    except Exception:
        return "mv"
    return "digital" if peak > 20 else "mv"


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
            "num_leads": meta["num_leads"],
            "lead_names": meta["lead_names"],
            "sampling_freq": meta["sampling_freq"],
            "num_samples": meta["num_samples"],
            "adc_gains": meta.get("adc_gains"),
            "adc_baselines": meta.get("adc_baselines"),
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
        norm_mode = record.get("_norm_mode", DEFAULT_NORM_MODE)
        bandpass = record.get("_bandpass", True)
        window_position = int(record.get("_window_position", 0))
        windows_max = int(record.get("_windows_max", 1))
        raw = load_mat_signal(record["mat_path"], record.get("num_leads", NUM_LEADS))
        full = preprocess_ecg_array(
            raw,
            sampling_rate=record["sampling_freq"],
            lead_names=record["lead_names"],
            target_fs=TARGET_FS,
            target_length=None,
            normalize=True,
            norm_mode=norm_mode,
            adc_gains=record.get("adc_gains"),
            adc_baselines=record.get("adc_baselines"),
            units="digital",
            bandpass=bool(bandpass),
        )
        starts = compute_window_starts(full.shape[0], WINDOW_LENGTH, windows_max)
        start = starts[min(window_position, len(starts) - 1)]
        piece = full[start:start + WINDOW_LENGTH, :]
        if piece.shape[0] < WINDOW_LENGTH:
            piece = np.pad(piece, ((0, WINDOW_LENGTH - piece.shape[0]), (0, 0)), mode="constant")
        signal = np.asarray(piece, dtype=np.float32)
        return {
            "ok": True,
            "signal": signal,
            "label": record["label"],
            "age": record["age"],
            "sex": record["sex"],
            "source_db": record["source_db"],
            "record_name": record["record_name"],
            "window_position": window_position,
            "window_start": int(start),
            "full_length": int(full.shape[0]),
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


def _expand_windows_for_record(record: dict, windows_max: int) -> list[dict]:
    """Expande un registro en 1..K specs (una por ventana de entrenamiento).

    El número de ventanas se estima desde el header (muestras * 500/fs).
    Registros de 10 s siempre producen 1 spec (ventana centrada).
    """
    windows_max = int(windows_max)
    try:
        n_target = int(round(float(record["num_samples"]) * TARGET_FS / float(record["sampling_freq"])))
    except Exception:
        n_target = WINDOW_LENGTH
    n_windows = len(compute_window_starts(n_target, WINDOW_LENGTH, windows_max))
    specs = []
    for position in range(n_windows):
        spec = dict(record)
        spec["_window_position"] = position
        spec["_windows_max"] = windows_max
        specs.append(spec)
    return specs


def _write_string_attr(h5_file, name: str, values: list[str]) -> None:
    import h5py
    dtype = h5py.string_dtype(encoding="utf-8")
    h5_file.attrs.create(name, np.asarray(values, dtype=dtype))


def _write_hdf5_attributes(
    h5_file,
    split_name: str,
    n_requested: int,
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
    windows_max: int = 1,
) -> None:
    h5_file.attrs["dataset"] = "PhysioNet/CinC Challenge 2020"
    h5_file.attrs["label_schema"] = LABEL_SCHEMA
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
    h5_file.attrs["norm_mode"] = str(norm_mode)
    h5_file.attrs["bandpass_filter"] = bool(bandpass)
    h5_file.attrs["windows_max"] = int(windows_max)
    if norm_mode == "physical":
        h5_file.attrs["normalization"] = (
            f"physical millivolts via header gain/baseline + bandpass {BANDPASS_LO_HZ}-{BANDPASS_HI_HZ} Hz + clip ±{PHYSICAL_CLIP_MV} mV"
            if bandpass else f"physical millivolts via header gain/baseline + clip ±{PHYSICAL_CLIP_MV} mV"
        )
    elif norm_mode == "global_zscore":
        h5_file.attrs["normalization"] = "per-record global z-score (all leads jointly) before padding"
    else:
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
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
    windows_max: int = 1,
) -> dict:
    import h5py
    from tqdm import tqdm

    indices = np.asarray(list(indices), dtype=int)
    selected = [records[int(i)] for i in indices]
    # Expansión multi-ventana (train): registros largos aportan hasta
    # windows_max muestras de 10 s equiespaciadas con la misma etiqueta.
    expanded: list[dict] = []
    for record in selected:
        for spec in _expand_windows_for_record(record, windows_max):
            spec["_norm_mode"] = norm_mode
            spec["_bandpass"] = bool(bandpass)
            expanded.append(spec)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as h5:
        chunk_n = min(16, max(1, len(expanded)))
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
            chunks=(min(256, max(1, len(expanded))), NUM_CLASSES),
        )
        ages_ds = h5.create_dataset("ages", shape=(0,), maxshape=(None,), dtype="float32")
        sexes_ds = h5.create_dataset("sexes", shape=(0,), maxshape=(None,), dtype=string_dtype)
        sources_ds = h5.create_dataset("source_dbs", shape=(0,), maxshape=(None,), dtype=string_dtype)
        names_ds = h5.create_dataset("record_names", shape=(0,), maxshape=(None,), dtype=string_dtype)
        dx_ds = h5.create_dataset("dx_codes", shape=(0,), maxshape=(None,), dtype=string_dtype)
        matched_ds = h5.create_dataset("matched_classes", shape=(0,), maxshape=(None,), dtype=string_dtype)
        winpos_ds = h5.create_dataset("window_positions", shape=(0,), maxshape=(None,), dtype="int32")
        winstart_ds = h5.create_dataset("window_starts", shape=(0,), maxshape=(None,), dtype="int32")

        _write_hdf5_attributes(h5, output_path.stem, len(selected), norm_mode, bandpass, windows_max)

        buffers = {
            "signals": [], "labels": [], "ages": [], "sexes": [],
            "source_dbs": [], "record_names": [], "dx_codes": [], "matched_classes": [],
            "window_positions": [], "window_starts": [],
        }
        n_ok, n_error = 0, 0
        error_examples = []

        def flush() -> None:
            nonlocal n_ok
            if not buffers["signals"]:
                return
            batch_size = len(buffers["signals"])
            old, new = n_ok, n_ok + batch_size
            for dataset in [signals_ds, labels_ds, ages_ds, sexes_ds, sources_ds, names_ds, dx_ds, matched_ds, winpos_ds, winstart_ds]:
                dataset.resize(new, axis=0)
            signals_ds[old:new] = np.stack(buffers["signals"]).astype(np.float32, copy=False)
            labels_ds[old:new] = np.stack(buffers["labels"]).astype(np.float32, copy=False)
            ages_ds[old:new] = np.asarray(buffers["ages"], dtype=np.float32)
            sexes_ds[old:new] = buffers["sexes"]
            sources_ds[old:new] = buffers["source_dbs"]
            names_ds[old:new] = buffers["record_names"]
            dx_ds[old:new] = buffers["dx_codes"]
            matched_ds[old:new] = buffers["matched_classes"]
            winpos_ds[old:new] = np.asarray(buffers["window_positions"], dtype=np.int32)
            winstart_ds[old:new] = np.asarray(buffers["window_starts"], dtype=np.int32)
            n_ok = new
            for values in buffers.values():
                values.clear()

        if num_workers == 1:
            iterator = map(_process_single_record, expanded)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=num_workers)
            iterator = executor.map(_process_single_record, expanded, chunksize=chunksize)

        try:
            for result in tqdm(iterator, total=len(expanded), desc=f"Procesando {output_path.name}", unit="muestra"):
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
                buffers["window_positions"].append(result.get("window_position", 0))
                buffers["window_starts"].append(result.get("window_start", 0))
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
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
    train_windows_max: int = 4,
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
        "label_schema": LABEL_SCHEMA,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "classes": class_groups_as_records(),
        "excluded_groups": EXCLUDED_GROUPS,
        "target_fs": TARGET_FS,
        "window_seconds": WINDOW_SECONDS,
        "window_length": WINDOW_LENGTH,
        "num_leads": NUM_LEADS,
        "lead_order": STANDARD_LEAD_ORDER,
        "norm_mode": str(norm_mode),
        "bandpass_filter": bool(bandpass),
        "bandpass_hz": [BANDPASS_LO_HZ, BANDPASS_HI_HZ],
        "physical_clip_mv": PHYSICAL_CLIP_MV,
        "train_windows_max": int(train_windows_max),
        "normalization": (
            f"physical millivolts + bandpass + clip ±{PHYSICAL_CLIP_MV} mV" if norm_mode == "physical"
            else "per-record global z-score" if norm_mode == "global_zscore"
            else "per-lead z-score (legacy v1)"
        ),
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
    norm_mode: str = DEFAULT_NORM_MODE,
    bandpass: bool = True,
    train_windows_max: int = 4,
) -> dict:
    """Función programática usada por ``examples/cinc2020/build_datasets.py``."""
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    print("\n" + "=" * 80)
    print("PREPROCESAMIENTO CINC2020 — 12 CLASES AGRUPADAS")
    print("=" * 80)
    print(f"Esquema     : {LABEL_SCHEMA}")
    print(f"Dataset     : {data_dir}")
    print(f"Salida      : {output_dir}")
    print(f"Señal       : {NUM_LEADS} leads, {TARGET_FS} Hz, {WINDOW_LENGTH} muestras")
    print(f"Norm        : {norm_mode} (bandpass={bandpass})")
    print(f"Ventanas    : train x{train_windows_max} / val-test x1")
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
        write_reports(output_dir, records, counters, splits=None, drop_no_selected_labels=drop_no_selected_labels,
                      norm_mode=norm_mode, bandpass=bandpass, train_windows_max=train_windows_max)
        return {"records": len(records), "counters": counters, "splits": None}

    splits = stratified_multilabel_split(labels, val_frac=val_frac, test_frac=test_frac, random_state=random_state)
    print("\nSplits:")
    for split_name, idx in splits.items():
        print(f"  {split_name:<5}: {len(idx):,}")

    write_reports(output_dir, records, counters, splits=splits, drop_no_selected_labels=drop_no_selected_labels,
                  norm_mode=norm_mode, bandpass=bandpass, train_windows_max=train_windows_max)

    processing = []
    for split_name in ["train", "val", "test"]:
        processing.append(process_split_to_hdf5(
            records=records,
            indices=splits[split_name],
            output_path=output_dir / f"{split_name}.h5",
            num_workers=workers,
            chunksize=chunksize,
            write_batch_size=write_batch_size,
            norm_mode=norm_mode,
            bandpass=bandpass,
            windows_max=train_windows_max if split_name == "train" else 1,
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
    parser.add_argument("--norm_mode", choices=["physical", "global_zscore", "per_lead_zscore"],
                        default=DEFAULT_NORM_MODE,
                        help="Normalización: physical=mV+clip (v2), global_zscore, per_lead_zscore (legacy v1)")
    parser.add_argument("--bandpass", dest="bandpass", action="store_true", default=True,
                        help="Aplica pasa-banda 0.5-50 Hz en modo physical (defecto: activado)")
    parser.add_argument("--no-bandpass", dest="bandpass", action="store_false",
                        help="Desactiva el pasa-banda")
    parser.add_argument("--train_windows_max", type=int, default=4,
                        help="Ventanas de 10 s por registro largo en train (val/test siempre 1 centrada)")
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
        norm_mode=args.norm_mode,
        bandpass=args.bandpass,
        train_windows_max=args.train_windows_max,
    )


if __name__ == "__main__":
    freeze_support()
    main()
