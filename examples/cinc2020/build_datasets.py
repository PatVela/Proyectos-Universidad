"""Construye HDF5 train/val/test para CINC2020-12.

Ejemplo:
    python examples/cinc2020/build_datasets.py \
      --data_dir dataset2020 \
      --output_dir data/cinc2020_12 \
      --workers 6
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from ecg.load import main


if __name__ == "__main__":
    main()
