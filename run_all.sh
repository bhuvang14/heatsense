#!/usr/bin/env bash
# HeatSense: run the whole pipeline in order.
# Uses .venv/bin/python when present, otherwise $PYTHON or python3.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; fi
echo "[run_all] using $($PY --version 2>&1)"

# Only regenerate synthetic data when the existing grid is missing or synthetic,
# so a real GEE export in data/grid_features.csv is never overwritten.
regen=1
if [ -f data/grid_features.csv ]; then
  src="$($PY - <<'PY' 2>/dev/null || true
import pandas as pd
print(pd.read_csv("data/grid_features.csv", usecols=["data_source"], nrows=1).data_source.iloc[0])
PY
)"
  if [ -n "$src" ] && [ "$src" != "synthetic" ]; then
    regen=0
    echo "[run_all] keeping existing grid_features.csv (data_source=$src)"
  fi
fi

if [ "$regen" = "1" ]; then
  "$PY" -m pipeline.synthetic
fi

"$PY" -m pipeline.features
"$PY" -m pipeline.model
"$PY" -m pipeline.cooling
"$PY" -m pipeline.priority
"$PY" -m pipeline.summaries
"$PY" -m pipeline.outputs

echo "[run_all] done. Start the app with: $PY app/app.py"
