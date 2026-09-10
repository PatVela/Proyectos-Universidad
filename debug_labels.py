# -*- coding: utf-8 -*-
"""Inspección rápida del mapeo SNOMED -> 12 clases agrupadas."""

from __future__ import annotations

import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from ecg.load import CLASS_NAMES, CODE_TO_CLASS_NAME, codes_to_vector, parse_header


def main():
    hea_files = glob.glob(os.path.join("dataset2020", "**", "*.hea"), recursive=True)

    print("=" * 70)
    print("DEBUG DE ETIQUETAS CINC2020 — 12 CLASES AGRUPADAS")
    print("=" * 70)
    print(f"Clases: {len(CLASS_NAMES)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i:02d}: {name}")

    print(f"\nHeaders encontrados: {len(hea_files)}")

    total_codes = 0
    matched_codes = 0
    records_with_any = 0

    for hea_path in hea_files[:20]:
        meta = parse_header(hea_path)
        y = codes_to_vector(meta["dx_codes"])
        matched = [(code, CODE_TO_CLASS_NAME[code]) for code in meta["dx_codes"] if code in CODE_TO_CLASS_NAME]
        unmatched = [code for code in meta["dx_codes"] if code not in CODE_TO_CLASS_NAME]

        total_codes += len(meta["dx_codes"])
        matched_codes += len(matched)
        records_with_any += int(y.sum() > 0)

        print("\n" + "-" * 70)
        print(f"Registro     : {meta['record_name']}")
        print(f"Header       : {hea_path}")
        print(f"Dx originales: {meta['dx_codes']}")
        print(f"Matched      : {matched}")
        print(f"Unmatched    : {unmatched}")
        print(f"Vector 12    : {y.astype(int).tolist()}")

    print("\n" + "=" * 70)
    print(f"Códigos totales inspeccionados : {total_codes}")
    print(f"Códigos mapeados a 12 clases   : {matched_codes}")
    print(f"Registros con >=1 clase        : {records_with_any}")
    if total_codes > 0:
        print(f"Cobertura por código           : {100 * matched_codes / total_codes:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
