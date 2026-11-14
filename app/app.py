"""HeatSense Flask backend: static UI, precomputed data files, and /whatif.

All heavy work is precomputed; at startup this only loads JSON/CSV from data/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import config  # noqa: E402
from pipeline.cooling import whatif  # noqa: E402

app = Flask(__name__, static_folder="static", static_url_path="/static")

DATA_DIR = config.DATA_DIR
ALLOWED_DATA_FILES = {p.name for p in DATA_DIR.glob("*.json")} | {
    p.name for p in DATA_DIR.glob("*.geojson")
} | {p.name for p in DATA_DIR.glob("*.csv")}  # flat data/ only, no traversal


def load_state() -> dict:
    """Load everything the UI needs once at startup."""
    meta_path = DATA_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing - run ./run_all.sh first")
    state = {
        "meta": json.loads(meta_path.read_text()),
        "metrics": json.loads((DATA_DIR / "model_metrics.json").read_text()),
        "top10_hottest": json.loads((DATA_DIR / "top10_hottest.json").read_text()),
        "top10_planting": json.loads((DATA_DIR / "top10_planting.json").read_text()),
        "cooling": pd.read_csv(config.COOLING_CSV, dtype={"ward_id": str}),
    }
    return state


STATE = load_state()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/data/<path:filename>")
def data_file(filename: str):
    if filename not in ALLOWED_DATA_FILES:
        return jsonify({"error": f"file not available: {filename}"}), 404
    return send_from_directory(DATA_DIR, filename)


@app.get("/whatif")
def whatif_route():
    ward = (request.args.get("ward") or "").strip()
    if not ward:
        return jsonify({"error": "query parameter 'ward' is required"}), 400

    raw = request.args.get("dndvi", "0.1")
    try:
        dndvi = float(raw)
    except (TypeError, ValueError):
        return jsonify({"error": f"dndvi must be a number, got {raw!r}"}), 400
    if dndvi < 0:
        return jsonify({"error": "dndvi must be >= 0"}), 400

    try:
        result = whatif(ward, dndvi)
    except KeyError:
        return jsonify({"error": f"unknown ward: {ward}"}), 404
    result["clamped"] = dndvi > 0.5
    return jsonify(result)


def main() -> None:
    meta = STATE["meta"]
    print(f"[app] data_source={meta['data_source']} wards={meta['n_wards']} "
          f"R2={meta['overall_r2']}")
    if meta["data_source"] == "synthetic":
        print("[app] DEMO MODE: synthetic sample data is being served")
    # Default is 5000; PORT lets you dodge macOS AirPlay Receiver, which also
    # listens on 5000. e.g. `PORT=5001 python app/app.py`
    port = int(os.environ.get("PORT", "5000"))
    print(f"[app] open http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
