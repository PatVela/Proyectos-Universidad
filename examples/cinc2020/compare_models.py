"""Compara resultados ResNet vs CNN convencional en CINC2020-12.

Ejemplo:
    python examples/cinc2020/compare_models.py \
      --resnet results/cinc2020_12_resnet \
      --cnn results/cinc2020_12_cnn \
      --output results/cinc2020_12/model_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_global_metrics(results_dir: Path, model_name: str) -> dict:
    metrics_path = results_dir / "metrics_global.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No existe {metrics_path}")
    df = pd.read_csv(metrics_path)
    test = df[df["split"] == "test"]
    if test.empty:
        test = df.tail(1)
    row = test.iloc[0].to_dict()
    row["model"] = model_name
    row["results_dir"] = str(results_dir)
    return row


def main():
    parser = argparse.ArgumentParser(description="Compara métricas test de ResNet y CNN convencional")
    parser.add_argument("--resnet", required=True, help="Directorio de resultados de ResNet")
    parser.add_argument("--cnn", required=True, help="Directorio de resultados de CNN convencional")
    parser.add_argument("--output", default="results/cinc2020_12/model_comparison.csv")
    args = parser.parse_args()

    rows = [
        load_global_metrics(Path(args.resnet), "ResNet-34"),
        load_global_metrics(Path(args.cnn), "CNN convencional equivalente"),
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nComparación guardada en: {out}")


if __name__ == "__main__":
    main()
