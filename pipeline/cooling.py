"""Local cooling efficiency: OLS slope of LST on NDVI, per ward.

A negative slope = more vegetation goes with cooler surface temperature.
This is a CORRELATIONAL association, not a measured cooling effect: greening a
spot may cool less (or more) than the cross-sectional slope implies.

Fallback rule (per brief): use the city-wide slope when a ward has
n < 100 cells, R2 < 0.10, or a non-physical slope >= 0.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.features import load_grid

MIN_PIXELS = 100
MIN_R2 = 0.10
MAX_DNDVI = 0.5
LABEL = "Correlational scenario, not a causal or field-measured result."

_SLOPES: pd.DataFrame | None = None  # lazy cache for whatif()


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for y ~ x."""
    if len(x) < 3 or np.std(x) < 1e-9:
        return 0.0, float(np.mean(y)), 0.0
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def fit_wards(df: pd.DataFrame, city_slope: float, city_r2: float) -> pd.DataFrame:
    rows = []
    for ward_id, g in df.groupby("ward_id"):
        slope, intercept, r2 = ols(g["ndvi"].to_numpy(), g["lst"].to_numpy())
        n = len(g)
        fallback = bool(n < MIN_PIXELS or r2 < MIN_R2 or slope >= 0)
        rows.append(
            {
                "ward_id": str(ward_id),
                "ward_name": str(g["ward_name"].iloc[0]),
                "slope": round(slope, 4),
                "intercept": round(intercept, 3),
                "r2": round(r2, 4),
                "n": n,
                "slope_fallback": fallback,
                "slope_used": round(city_slope if fallback else slope, 4),
                "city_slope": round(city_slope, 4),
                "city_r2": round(city_r2, 4),
            }
        )
    out = pd.DataFrame(rows).sort_values("ward_id").reset_index(drop=True)
    return out


def load_slopes() -> pd.DataFrame:
    """Load (and memoise) data/cooling_by_ward.csv."""
    global _SLOPES
    if _SLOPES is None:
        if not config.COOLING_CSV.exists():
            raise FileNotFoundError(
                f"{config.COOLING_CSV} not found - run pipeline/cooling.py first"
            )
        _SLOPES = pd.read_csv(config.COOLING_CSV, dtype={"ward_id": str})
    return _SLOPES


def whatif(ward_id: str, dndvi: float) -> dict:
    """Predicted surface cooling for an NDVI increase in one ward.

    cooling_C = -(slope_used * dndvi); positive means cooling.
    dndvi is clamped to [0, 0.5].
    """
    slopes = load_slopes()
    row = slopes[slopes["ward_id"] == str(ward_id)]
    if row.empty:
        raise KeyError(f"unknown ward_id: {ward_id}")
    r = row.iloc[0]

    dndvi = float(dndvi)
    clamped = float(min(max(dndvi, 0.0), MAX_DNDVI))
    cooling = -(float(r["slope_used"]) * clamped)
    return {
        "ward_id": str(r["ward_id"]),
        "ward_name": str(r["ward_name"]),
        "dndvi": round(clamped, 3),
        "predicted_cooling_c": round(cooling, 2),
        "slope": round(float(r["slope_used"]), 4),
        "slope_r2": round(float(r["r2"] if not r["slope_fallback"] else r["city_r2"]), 4),
        "slope_fallback": bool(r["slope_fallback"]),
        "label": LABEL,
    }


def main() -> None:
    df, _ = load_grid()
    city_slope, _, city_r2 = ols(df["ndvi"].to_numpy(), df["lst"].to_numpy())
    print(f"[cooling] city-wide slope {city_slope:.3f} C per +1.0 NDVI (R2 {city_r2:.3f}, "
          f"n={len(df)})")

    slopes = fit_wards(df, city_slope, city_r2)
    slopes.to_csv(config.COOLING_CSV, index=False)

    n_fb = int(slopes["slope_fallback"].sum())
    print(f"[cooling] wards: {len(slopes)}, fallback {n_fb} "
          f"({n_fb / len(slopes) * 100:.1f}%), local slope range "
          f"{slopes.loc[~slopes.slope_fallback, 'slope'].min():.2f} to "
          f"{slopes.loc[~slopes.slope_fallback, 'slope'].max():.2f} C per unit NDVI")

    metrics = json.loads(config.METRICS_JSON.read_text())
    metrics.update(
        {
            "city_ndvi_slope_c_per_unit": round(city_slope, 4),
            "city_ndvi_slope_r2": round(city_r2, 4),
            "cooling_fallback_rule": (
                f"city-wide slope when n<{MIN_PIXELS} cells, R2<{MIN_R2}, or slope>=0"
            ),
            "cooling_fallback_wards": n_fb,
            "cooling_label": LABEL,
        }
    )
    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2))

    demo_ward = slopes.sort_values("n", ascending=False).iloc[0]["ward_id"]
    print("[cooling] example:", json.dumps(whatif(demo_ward, 0.1)))
    print(f"[cooling] wrote {config.COOLING_CSV.name} ({len(slopes)} wards) "
          f"and updated {config.METRICS_JSON.name}")


if __name__ == "__main__":
    sys.exit(main())
