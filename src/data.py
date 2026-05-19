"""data.py — re-exports the active feature pipeline.

We had two iterations:
  - data_v1.py: 14 features (the original PoC)
  - data_v2.py: 34 features (per-90 stats, top-tier minutes, club quality, career shape)

v2 won the experiment in scripts/experiment.py — it's the active pipeline.
The v1 module is kept for historical comparison only.

This file re-exports the v2 names so the rest of the project (scripts/main.py,
notebooks, the Streamlit app, etc.) doesn't need to change imports.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_v2 import (  # noqa: F401, E402
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET,
    _load_appearances,
    _load_players,
    load_dataset_split,
)
