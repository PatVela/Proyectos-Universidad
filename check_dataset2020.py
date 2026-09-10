# -*- coding: utf-8 -*-

"""
check_dataset2020.py

Valida los archivos HDF5 generados por examples/cinc2020/build_datasets.py
usando ecg/load.py.

Estructura esperada:

    proyecto/
    ├── dataset2020/
    ├── data/cinc2020_12/
    ├── ecg/
    │   ├── load.py
    │   └── ...
    └── check_dataset2020.py

Comprobaciones:

    - existencia de HDF5
    - shapes
    - número de registros
    - número de clases
    - orden de derivaciones
    - frecuencia de muestreo
    - longitud de ventana
    - NaN / Inf
    - estadísticas de señal
    - registros completamente constantes
    - registros con amplitud extremadamente baja
    - labels binarias {0,1}
    - positivos por clase
    - clases sin positivos
    - registros sin ninguna etiqueta
    - labels por registro
    - nombres duplicados
    - distribución por base
    - distribución de sexo
    - edades válidas / inválidas
"""

import os
import h5py
import numpy as np


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cinc2020_12"
)

EXPECTED_FS = 500
EXPECTED_LENGTH = 5000
EXPECTED_LEADS_COUNT = 12
EXPECTED_CLASSES = 12

EXPECTED_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

EXPECTED_DATASETS = [
    "signals",
    "labels",
    "ages",
    "sexes",
    "source_dbs",
    "record_names",
]

HDF5_FILES = [
    "train.h5",
    "val.h5",
    "test.h5",

]

# Número de registros de señal que se inspeccionan en profundidad.
DEFAULT_SIGNAL_SAMPLE_COUNT = 100

# Umbral para considerar una señal "prácticamente plana".
CONSTANT_STD_THRESHOLD = 1e-8

# Umbral opcional para detectar señales con amplitud muy baja.
LOW_AMPLITUDE_STD_THRESHOLD = 1e-3

# Rango fisiológico básico para detectar metadata sospechosa.
# No estamos eliminando edades aquí; solo las reportamos.
MIN_REASONABLE_AGE = 0
MAX_REASONABLE_AGE = 120


# =====================================================================
# UTILIDADES
# =====================================================================

def decode_string(value):
    """
    Convierte strings de HDF5 a Python str.
    """

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace"
        )

    return str(value)


def print_header(title):
    """
    Imprime encabezado visual.
    """

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# =====================================================================
# VALIDACIÓN DE UN ARCHIVO
# =====================================================================

def validate_h5(
    path,
    signal_sample_count=DEFAULT_SIGNAL_SAMPLE_COUNT
):
    """
    Valida un archivo HDF5 completo.

    Returns:
        True  -> estructura correcta sin errores críticos.
        False -> se encontró al menos un error crítico.
    """

    print_header(
        f"ARCHIVO: {os.path.basename(path)}"
    )

    if not os.path.exists(path):

        print(
            "ERROR: archivo no encontrado."
        )

        return False

    errors = 0
    warnings = 0

    try:

        with h5py.File(
            path,
            "r"
        ) as f:

            # =====================================================
            # DATASETS
            # =====================================================

            print()
            print("DATASETS")

            for dataset_name in EXPECTED_DATASETS:

                if dataset_name not in f:

                    print(
                        f"ERROR: falta "
                        f"'{dataset_name}'."
                    )

                    errors += 1

            if errors > 0:

                return False

            signals = f["signals"]
            labels = f["labels"]
            ages = f["ages"]
            sexes = f["sexes"]
            source_dbs = f["source_dbs"]
            record_names = f["record_names"]

            # =====================================================
            # SHAPES
            # =====================================================

            print()
            print("SHAPES")

            print(
                f"  signals       : {signals.shape}"
            )

            print(
                f"  labels        : {labels.shape}"
            )

            print(
                f"  ages          : {ages.shape}"
            )

            print(
                f"  sexes         : {sexes.shape}"
            )

            print(
                f"  source_dbs    : {source_dbs.shape}"
            )

            print(
                f"  record_names  : {record_names.shape}"
            )

            # -----------------------------------------------------
            # Número de registros
            # -----------------------------------------------------

            if signals.ndim != 3:

                print(
                    "ERROR: signals debe ser "
                    "tridimensional."
                )

                errors += 1

                return False

            n_records = signals.shape[0]

            # -----------------------------------------------------
            # Shape de signals
            # -----------------------------------------------------

            expected_signal_shape = (
                n_records,
                EXPECTED_LENGTH,
                EXPECTED_LEADS_COUNT
            )

            if (
                signals.shape
                != expected_signal_shape
            ):

                print(
                    "ERROR: shape de signals "
                    "incorrecta."
                )

                print(
                    f"  Esperada: "
                    f"{expected_signal_shape}"
                )

                print(
                    f"  Actual:   "
                    f"{signals.shape}"
                )

                errors += 1

            # -----------------------------------------------------
            # Shape de labels
            # -----------------------------------------------------

            if labels.ndim != 2:

                print(
                    "ERROR: labels debe ser "
                    "bidimensional."
                )

                errors += 1

            else:

                expected_label_shape = (
                    n_records,
                    EXPECTED_CLASSES
                )

                if (
                    labels.shape
                    != expected_label_shape
                ):

                    print(
                        "ERROR: shape de labels "
                        "incorrecta."
                    )

                    print(
                        f"  Esperada: "
                        f"{expected_label_shape}"
                    )

                    print(
                        f"  Actual:   "
                        f"{labels.shape}"
                    )

                    errors += 1

            # -----------------------------------------------------
            # Metadata
            # -----------------------------------------------------

            metadata_datasets = {
                "ages": ages,
                "sexes": sexes,
                "source_dbs": source_dbs,
                "record_names": record_names,
            }

            for name, dataset in metadata_datasets.items():

                if len(dataset) != n_records:

                    print(
                        f"ERROR: '{name}' tiene "
                        f"{len(dataset)} elementos; "
                        f"se esperaban {n_records}."
                    )

                    errors += 1

            # =====================================================
            # ATTRIBUTES
            # =====================================================

            print()
            print("ATTRIBUTES")

            classes_attr = f.attrs.get(
                "classes",
                None
            )

            sampling_rate = f.attrs.get(
                "sampling_rate",
                None
            )

            window_seconds = f.attrs.get(
                "window_seconds",
                None
            )

            window_length = f.attrs.get(
                "window_length",
                None
            )

            num_leads = f.attrs.get(
                "num_leads",
                None
            )

            lead_order = f.attrs.get(
                "lead_order",
                None
            )

            print(
                f"  sampling_rate : {sampling_rate}"
            )

            print(
                f"  window_seconds: {window_seconds}"
            )

            print(
                f"  window_length : {window_length}"
            )

            print(
                f"  num_leads     : {num_leads}"
            )

            print(
                f"  lead_order    : {lead_order}"
            )

            # -----------------------------------------------------
            # Clases
            # -----------------------------------------------------

            classes = None

            if classes_attr is None:

                print(
                    "ERROR: falta atributo "
                    "'classes'."
                )

                errors += 1

            else:

                classes = [
                    decode_string(x)
                    for x in classes_attr
                ]

                print(
                    f"  clases        : "
                    f"{len(classes)}"
                )

                for i, class_name in enumerate(
                    classes
                ):

                    print(
                        f"    {i:02d}: "
                        f"{class_name}"
                    )

                if len(classes) != EXPECTED_CLASSES:

                    print(
                        f"ERROR: se esperaban "
                        f"{EXPECTED_CLASSES} clases."
                    )

                    errors += 1

            # -----------------------------------------------------
            # Leads
            # -----------------------------------------------------

            if lead_order is None:

                print(
                    "ERROR: falta atributo "
                    "'lead_order'."
                )

                errors += 1

            else:

                actual_lead_order = [
                    x.strip()
                    for x in decode_string(
                        lead_order
                    ).split(",")
                ]

                if (
                    actual_lead_order
                    != EXPECTED_LEADS
                ):

                    print(
                        "ERROR: orden de derivaciones "
                        "incorrecto."
                    )

                    print(
                        f"  Esperado: "
                        f"{EXPECTED_LEADS}"
                    )

                    print(
                        f"  Actual:   "
                        f"{actual_lead_order}"
                    )

                    errors += 1

            # -----------------------------------------------------
            # Parámetros
            # -----------------------------------------------------

            if sampling_rate != EXPECTED_FS:

                print(
                    f"ERROR: sampling_rate = "
                    f"{sampling_rate}; "
                    f"esperado = {EXPECTED_FS}."
                )

                errors += 1

            if window_length != EXPECTED_LENGTH:

                print(
                    f"ERROR: window_length = "
                    f"{window_length}; "
                    f"esperado = {EXPECTED_LENGTH}."
                )

                errors += 1

            if num_leads != EXPECTED_LEADS_COUNT:

                print(
                    f"ERROR: num_leads = "
                    f"{num_leads}; "
                    f"esperado = "
                    f"{EXPECTED_LEADS_COUNT}."
                )

                errors += 1

            # =====================================================
            # SI NO HAY REGISTROS
            # =====================================================

            if n_records == 0:

                print(
                    "ERROR: el archivo "
                    "no contiene registros."
                )

                errors += 1

                return False

            # =====================================================
            # INSPECCIÓN DE SEÑALES
            # =====================================================

            print()
            print("SEÑALES")

            sample_count = min(
                signal_sample_count,
                n_records
            )

            # Elegimos registros distribuidos
            # a lo largo del archivo.
            sample_indices = np.linspace(
                0,
                n_records - 1,
                sample_count,
                dtype=int
            )

            sample_signals = signals[
                sample_indices
            ]

            print(
                f"  Registros inspeccionados: "
                f"{sample_count}/{n_records}"
            )

            # -----------------------------------------------------
            # NaN / Inf
            # -----------------------------------------------------

            nan_count = int(
                np.isnan(
                    sample_signals
                ).sum()
            )

            inf_count = int(
                np.isinf(
                    sample_signals
                ).sum()
            )

            print(
                f"  NaN : {nan_count}"
            )

            print(
                f"  Inf : {inf_count}"
            )

            if nan_count > 0:

                print(
                    "ERROR: se detectaron NaN."
                )

                errors += 1

            if inf_count > 0:

                print(
                    "ERROR: se detectaron Inf."
                )

                errors += 1

            # -----------------------------------------------------
            # Estadísticas generales
            # -----------------------------------------------------

            signal_min = float(
                sample_signals.min()
            )

            signal_max = float(
                sample_signals.max()
            )

            signal_mean = float(
                sample_signals.mean()
            )

            signal_std = float(
                sample_signals.std()
            )

            print(
                f"  mínimo : {signal_min:.6f}"
            )

            print(
                f"  máximo : {signal_max:.6f}"
            )

            print(
                f"  media  : {signal_mean:.6f}"
            )

            print(
                f"  std    : {signal_std:.6f}"
            )

            # -----------------------------------------------------
            # Registros constantes
            # -----------------------------------------------------

            constant_records = 0
            constant_record_names = []

            for local_idx, signal in enumerate(
                sample_signals
            ):

                lead_std = np.std(
                    signal,
                    axis=0
                )

                if np.all(
                    lead_std
                    < CONSTANT_STD_THRESHOLD
                ):

                    constant_records += 1

                    original_idx = (
                        sample_indices[
                            local_idx
                        ]
                    )

                    record_name = (
                        decode_string(
                            record_names[
                                original_idx
                            ]
                        )
                    )

                    constant_record_names.append(
                        record_name
                    )

            print(
                f"  Registros constantes: "
                f"{constant_records}/{sample_count}"
            )

            if constant_record_names:

                print(
                    "  Registros constantes encontrados:"
                )

                for record_name in (
                    constant_record_names
                ):

                    print(
                        f"    - {record_name}"
                    )

                warnings += 1

            # -----------------------------------------------------
            # Registros con amplitud muy baja
            # -----------------------------------------------------

            low_amplitude_records = 0
            low_amplitude_names = []

            for local_idx, signal in enumerate(
                sample_signals
            ):

                lead_std = np.std(
                    signal,
                    axis=0
                )

                if np.all(
                    lead_std
                    < LOW_AMPLITUDE_STD_THRESHOLD
                ):

                    low_amplitude_records += 1

                    original_idx = (
                        sample_indices[
                            local_idx
                        ]
                    )

                    record_name = (
                        decode_string(
                            record_names[
                                original_idx
                            ]
                        )
                    )

                    low_amplitude_names.append(
                        record_name
                    )

            print(
                f"  Baja amplitud: "
                f"{low_amplitude_records}/"
                f"{sample_count}"
            )

            if low_amplitude_names:

                print(
                    "  Registros de baja amplitud:"
                )

                for record_name in (
                    low_amplitude_names
                ):

                    print(
                        f"    - {record_name}"
                    )

            # =====================================================
            # ETIQUETAS
            # =====================================================

            print()
            print("ETIQUETAS")

            # 12 x N normalmente es pequeño
            # comparado con las señales.
            label_array = labels[:]

            unique_values = np.unique(
                label_array
            )

            print(
                f"  Valores encontrados: "
                f"{unique_values}"
            )

            invalid_values = unique_values[
                ~np.isin(
                    unique_values,
                    [0.0, 1.0]
                )
            ]

            if len(invalid_values) > 0:

                print(
                    "ERROR: labels contiene "
                    f"valores distintos de 0/1: "
                    f"{invalid_values}"
                )

                errors += 1

            # -----------------------------------------------------
            # Positivos
            # -----------------------------------------------------

            positives = label_array.sum(
                axis=0
            )

            print()
            print(
                "  Positivos por clase:"
            )

            for i, count in enumerate(
                positives
            ):

                class_name = (
                    classes[i]
                    if classes is not None
                    and i < len(classes)
                    else f"class_{i}"
                )

                percentage = (
                    100.0
                    * count
                    / n_records
                )

                print(
                    f"    {i:02d} "
                    f"{class_name:10s}: "
                    f"{int(count):7d} "
                    f"({percentage:6.2f} %)"
                )

                if count == 0:

                    print(
                        f"      WARNING: "
                        f"clase {class_name} "
                        f"sin positivos."
                    )

                    warnings += 1

            # -----------------------------------------------------
            # Registros sin etiquetas
            # -----------------------------------------------------

            labels_per_record = (
                label_array.sum(
                    axis=1
                )
            )

            no_label = (
                labels_per_record == 0
            )

            n_no_label = int(
                no_label.sum()
            )

            print()
            print(
                "  Registros sin ninguna "
                "clase puntuable: "
                f"{n_no_label} "
                f"({100*n_no_label/n_records:.2f} %)"
            )

            print()
            print(
                "  Labels por registro:"
            )

            print(
                f"    media  : "
                f"{labels_per_record.mean():.3f}"
            )

            print(
                f"    mínimo : "
                f"{labels_per_record.min():.0f}"
            )

            print(
                f"    máximo : "
                f"{labels_per_record.max():.0f}"
            )

            # =====================================================
            # RECORD NAMES
            # =====================================================

            print()
            print("RECORD NAMES")

            names = [
                decode_string(x)
                for x in record_names[:]
            ]

            unique_names = len(
                set(names)
            )

            duplicate_count = (
                len(names)
                - unique_names
            )

            print(
                f"  Total      : {len(names)}"
            )

            print(
                f"  Únicos     : {unique_names}"
            )

            print(
                f"  Duplicados : {duplicate_count}"
            )

            if duplicate_count > 0:

                print(
                    "WARNING: existen "
                    "record_names duplicados."
                )

                warnings += 1

                # Mostrar hasta 10 duplicados
                counts = {}

                for name in names:

                    counts[name] = (
                        counts.get(name, 0)
                        + 1
                    )

                duplicates = [
                    name
                    for name, count
                    in counts.items()
                    if count > 1
                ]

                for name in duplicates[:10]:

                    print(
                        f"    - {name}"
                    )

            # =====================================================
            # BASES DE DATOS
            # =====================================================

            print()
            print("BASES DE DATOS")

            db_names = [
                decode_string(x)
                for x in source_dbs[:]
            ]

            unique_dbs, db_counts = (
                np.unique(
                    db_names,
                    return_counts=True
                )
            )

            for db, count in zip(
                unique_dbs,
                db_counts
            ):

                percentage = (
                    100.0
                    * count
                    / n_records
                )

                print(
                    f"  {db:25s}: "
                    f"{count:7d} "
                    f"({percentage:6.2f} %)"
                )

            # =====================================================
            # SEXO
            # =====================================================

            print()
            print("SEXO")

            sexes_list = [
                decode_string(x)
                for x in sexes[:]
            ]

            unique_sexes, sex_counts = (
                np.unique(
                    sexes_list,
                    return_counts=True
                )
            )

            for sex, count in zip(
                unique_sexes,
                sex_counts
            ):

                percentage = (
                    100.0
                    * count
                    / n_records
                )

                print(
                    f"  {sex:10s}: "
                    f"{count:7d} "
                    f"({percentage:6.2f} %)"
                )

            # =====================================================
            # EDAD
            # =====================================================

            print()
            print("EDAD")

            ages_array = np.asarray(
                ages[:],
                dtype=np.float32
            )

            valid_numeric = np.isfinite(
                ages_array
            )

            n_numeric = int(
                valid_numeric.sum()
            )

            print(
                f"  Numéricas: "
                f"{n_numeric}/{n_records}"
            )

            # -----------------------------------------------------
            # Edades dentro de rango razonable
            # -----------------------------------------------------

            reasonable = (
                valid_numeric
                & (
                    ages_array
                    >= MIN_REASONABLE_AGE
                )
                & (
                    ages_array
                    <= MAX_REASONABLE_AGE
                )
            )

            n_reasonable = int(
                reasonable.sum()
            )

            invalid_age_mask = (
                valid_numeric
                & ~(
                    (
                        ages_array
                        >= MIN_REASONABLE_AGE
                    )
                    & (
                        ages_array
                        <= MAX_REASONABLE_AGE
                    )
                )
            )

            n_invalid_age = int(
                invalid_age_mask.sum()
            )

            print(
                f"  En rango 0-120: "
                f"{n_reasonable}/{n_records}"
            )

            print(
                f"  Fuera de rango: "
                f"{n_invalid_age}"
            )

            if n_invalid_age > 0:

                warnings += 1

                invalid_values = (
                    ages_array[
                        invalid_age_mask
                    ]
                )

                print(
                    "  Valores de edad "
                    "sospechosos:"
                )

                unique_invalid = np.unique(
                    invalid_values
                )

                for value in (
                    unique_invalid[:20]
                ):

                    print(
                        f"    - {value}"
                    )

            if n_reasonable > 0:

                valid_ages = (
                    ages_array[
                        reasonable
                    ]
                )

                print(
                    f"  Mínimo válido : "
                    f"{valid_ages.min():.1f}"
                )

                print(
                    f"  Máximo válido : "
                    f"{valid_ages.max():.1f}"
                )

                print(
                    f"  Media válida  : "
                    f"{valid_ages.mean():.1f}"
                )

            # =====================================================
            # INFORMACIÓN ESPECÍFICA
            # =====================================================

            print()
            print("INFORMACIÓN DE ALMACENAMIENTO")

            print(
                f"  signals dtype : "
                f"{signals.dtype}"
            )

            print(
                f"  labels dtype  : "
                f"{labels.dtype}"
            )

            print(
                f"  compression   : "
                f"{signals.compression}"
            )

            print(
                f"  gzip level    : "
                f"{signals.compression_opts}"
            )

            # =====================================================
            # RESULTADO
            # =====================================================

            print()
            print("-" * 80)

            if errors == 0:

                print(
                    "RESULTADO: OK"
                )

            else:

                print(
                    f"RESULTADO: "
                    f"{errors} ERROR(ES)"
                )

            if warnings > 0:

                print(
                    f"ADVERTENCIAS: "
                    f"{warnings}"
                )

            print("-" * 80)

    except OSError as e:

        print()
        print(
            f"ERROR al abrir HDF5: {e}"
        )

        return False

    except Exception as e:

        print()
        print(
            f"ERROR inesperado: {e}"
        )

        return False

    return errors == 0


# =====================================================================
# MAIN
# =====================================================================

def main():

    print_header(
        "VALIDACIÓN DATASET CINC2020"
    )

    print(
        f"Proyecto  : {PROJECT_ROOT}"
    )

    print(
        f"Directorio: {DATASET_DIR}"
    )

    print(
        f"Esperado  : "
        f"{EXPECTED_CLASSES} clases, "
        f"{EXPECTED_LEADS_COUNT} leads, "
        f"{EXPECTED_FS} Hz, "
        f"{EXPECTED_LENGTH} muestras"
    )

    results = {}

    # -------------------------------------------------------------
    # Validar archivos
    # -------------------------------------------------------------

    for filename in HDF5_FILES:

        path = os.path.join(
            DATASET_DIR,
            filename
        )

        results[filename] = validate_h5(
            path
        )

    # -------------------------------------------------------------
    # Resultado general
    # -------------------------------------------------------------

    print_header(
        "RESULTADO GENERAL"
    )

    all_ok = True

    for filename, ok in results.items():

        status = (
            "OK"
            if ok
            else "ERROR"
        )

        print(
            f"{filename:25s}: "
            f"{status}"
        )

        if not ok:

            all_ok = False

    print()

    if all_ok:

        print(
            "TODOS LOS HDF5 PASARON "
            "LAS COMPROBACIONES ESTRUCTURALES."
        )

    else:

        print(
            "HAY HDF5 QUE REQUIEREN REVISIÓN."
        )

    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()