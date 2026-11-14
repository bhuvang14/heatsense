"""Where a tree cools most for the most exposed people.

Candidate cells: plantable, hotter than their ward's 75th percentile LST, and
below the city's 40th percentile NDVI (bare, hot, ungreened ground).

Pixel score = local_cooling_efficiency x plantable_area x exposed_population,
with each factor min-max normalised to 0-1 across candidates first so that no
single factor dominates by units alone.

Ward score = SUM of its candidate pixel scores (rewards both intensity and the
number of opportunities in the ward; documented in README).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.cooling import LABEL, load_slopes
from pipeline.features import CELL_AREA_M2, load_grid

W_LST_PCT = 0.75
CITY_NDVI_PCT = 0.40
MIN_SPACING_M = 1000.0
NEIGHBOURHOOD_CELLS = 3  # 3x3 cells (~300 m) population smoothing
TOP_N = 10


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(1.0, index=s.index)
    return (s - lo) / (hi - lo)


def _haversine_m(lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def smooth_population(df: pd.DataFrame) -> pd.Series:
    """Mean population over a 3x3 (~300 m) neighbourhood of grid cells."""
    b = config.BBOX
    lat0 = (b["lat_min"] + b["lat_max"]) / 2
    dlat = config.GRID_SIZE_M_SYNTHETIC / 110_574.0
    dlon = config.GRID_SIZE_M_SYNTHETIC / (111_320.0 * np.cos(np.radians(lat0)))
    ix = np.round((df["lon"] - b["lon_min"]) / dlon).astype(int)
    iy = np.round((df["lat"] - b["lat_min"]) / dlat).astype(int)
    block = (ix // NEIGHBOURHOOD_CELLS).astype(str) + "_" + (iy // NEIGHBOURHOOD_CELLS).astype(str)
    return df.groupby(block)["population"].transform("mean")


def compute_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the candidate filter and score every candidate cell."""
    slopes = load_slopes()[["ward_id", "slope_used", "slope_fallback", "r2", "city_r2"]]
    ward_p75 = df.groupby("ward_id")["lst"].transform(lambda s: s.quantile(W_LST_PCT))
    city_ndvi_p40 = float(df["ndvi"].quantile(CITY_NDVI_PCT))

    mask = (df["plantable"] == 1) & (df["lst"] > ward_p75) & (df["ndvi"] < city_ndvi_p40)
    cand = df.loc[mask].copy()
    if cand.empty:
        raise RuntimeError("no candidate cells - check plantable/LST/NDVI thresholds")
    # Smoothed population is indexed like df, so attach it before the merge.
    cand["population_300m"] = smooth_population(df).loc[cand.index]
    cand = cand.merge(slopes, on="ward_id", how="left", suffixes=("", "_cool"))
    cand["cooling_eff"] = cand["slope_used"].abs()
    cand["plantable_area_m2"] = float(CELL_AREA_M2)
    cand["exposed_population"] = cand["population_300m"].fillna(cand["population"])

    cand["f_cooling"] = _minmax(cand["cooling_eff"])
    cand["f_area"] = _minmax(cand["plantable_area_m2"])
    cand["f_pop"] = _minmax(cand["exposed_population"])
    cand["score"] = cand["f_cooling"] * cand["f_area"] * cand["f_pop"]
    cand["expected_cooling_0_1"] = (-cand["slope_used"] * 0.1).round(2)
    return cand.sort_values("score", ascending=False).reset_index(drop=True)


def top_spots(cand: pd.DataFrame, k: int = TOP_N, min_spacing_m: float = MIN_SPACING_M) -> list[dict]:
    """Greedy top-k with a minimum great-circle spacing between picks."""
    picked: list[dict] = []
    for row in cand.itertuples(index=False):
        if picked:
            lons = np.array([p["lon"] for p in picked])
            lats = np.array([p["lat"] for p in picked])
            if (_haversine_m(row.lon, row.lat, lons, lats) < min_spacing_m).any():
                continue
        picked.append(
            {
                "cell_id": int(row.cell_id),
                "lon": round(float(row.lon), 5),
                "lat": round(float(row.lat), 5),
                "ward_id": str(row.ward_id),
                "ward_name": str(row.ward_name),
                "score": round(float(row.score), 4),
                "lst_c": round(float(row.lst), 2),
                "ndvi": round(float(row.ndvi), 3),
                "population_cell": int(row.population),
                "population_300m": round(float(row.exposed_population), 1),
                "expected_cooling_c_per_0_1_ndvi": round(float(row.expected_cooling_0_1), 2),
                "slope_used": round(float(row.slope_used), 3),
                "slope_fallback": bool(row.slope_fallback),
            }
        )
        if len(picked) == k:
            break
    return picked


def ward_scores(cand: pd.DataFrame) -> pd.DataFrame:
    """Ward priority = sum of candidate pixel scores."""
    out = (
        cand.groupby("ward_id", as_index=False)
        .agg(priority_score=("score", "sum"), n_candidates=("score", "size"))
        .sort_values("priority_score", ascending=False)
        .reset_index(drop=True)
    )
    out["priority_rank"] = np.arange(1, len(out) + 1)
    out["priority_score"] = out["priority_score"].round(2)
    return out


def equity_flags(wards: pd.DataFrame) -> pd.DataFrame:
    """Hot + dense + low-green wards (top/top/bottom quartile of ward values)."""
    hot = wards["lst_mean"] >= wards["lst_mean"].quantile(0.75)
    dense = wards["pop_density"] >= wards["pop_density"].quantile(0.75)
    low_green = wards["ndvi_mean"] <= wards["ndvi_mean"].quantile(0.25)
    out = wards[["ward_id"]].copy()
    out["equity_flag"] = (hot & dense & low_green)
    return out


def main() -> None:
    df, _ = load_grid()
    cand = compute_candidates(df)
    spots = top_spots(cand)
    wscores = ward_scores(cand)
    print(f"[priority] candidates: {len(cand)} cells in {cand['ward_id'].nunique()} wards "
          f"({len(cand) / len(df) * 100:.1f}% of all cells)")

    payload = {
        "data_source": str(df["data_source"].iloc[0]),
        "label": LABEL,
        "method": {
            "candidate_filter": (
                f"plantable=1 & lst > ward p{int(W_LST_PCT * 100)} & ndvi < city p"
                f"{int(CITY_NDVI_PCT * 100)}"
            ),
            "score": (
                "minmax(local |slope|) * minmax(plantable area) * minmax(population, "
                f"{NEIGHBOURHOOD_CELLS}x{NEIGHBOURHOOD_CELLS} cell mean)"
            ),
            "min_spacing_m": MIN_SPACING_M,
            "note": (
                "Plantable area is the full 100 m cell (10,000 m2) for every candidate, so "
                "that factor is constant here and does not change the ranking."
            ),
        },
        "spots": spots,
    }
    config.TOP10_PLANTING_JSON.write_text(json.dumps(payload, indent=2))

    wscores.to_csv(config.DATA_DIR / "ward_priority.csv", index=False)
    print(f"[priority] top ward scores:\n{wscores.head(5).to_string(index=False)}")
    print(f"[priority] top-{len(spots)} planting spots (>= {MIN_SPACING_M:.0f} m apart):")
    for i, s in enumerate(spots, start=1):
        print(f"  {i:2d}. {s['ward_name']:<24} {s['lst_c']:.1f} C  score {s['score']:.3f}  "
              f"cooling +0.1 NDVI ~{s['expected_cooling_c_per_0_1_ndvi']:.1f} C")
    print(f"[priority] wrote {config.TOP10_PLANTING_JSON.name} and ward_priority.csv")


if __name__ == "__main__":
    sys.exit(main())
