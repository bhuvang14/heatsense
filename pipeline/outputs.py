"""Assemble the final artefacts: wards.geojson, top-10 lists and app metadata."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd

from pipeline import config
from pipeline.priority import equity_flags
from pipeline.wards import load_wards


def _top_drivers(drivers: dict, ward_id: str) -> list[dict]:
    entry = drivers.get(str(ward_id), {})
    return [
        {
            "feature": d["feature"],
            "direction": d["direction"],
            "share": d["share"],
        }
        for d in entry.get("top_drivers", [])
    ]


def build() -> tuple[pd.DataFrame, dict]:
    wards = load_wards()
    agg = pd.read_csv(config.WARD_AGG_CSV, dtype={"ward_id": str})
    cool = pd.read_csv(config.COOLING_CSV, dtype={"ward_id": str})
    prio = pd.read_csv(config.DATA_DIR / "ward_priority.csv", dtype={"ward_id": str})
    drivers = json.loads(config.SHAP_WARD_JSON.read_text())["wards"]
    summaries = json.loads((config.DATA_DIR / "summaries.json").read_text())
    metrics = json.loads(config.METRICS_JSON.read_text())

    equity = equity_flags(agg)
    df = agg.merge(cool[["ward_id", "slope", "r2", "n", "slope_fallback", "slope_used"]],
                   on="ward_id", how="left")
    df = df.merge(prio[["ward_id", "priority_score", "priority_rank"]], on="ward_id", how="left")
    df = df.merge(equity, on="ward_id", how="left")

    # Wards with no planting candidates still appear, ranked at the bottom.
    df["priority_score"] = df["priority_score"].fillna(0.0)
    order = df.sort_values("priority_score", ascending=False)["ward_id"].tolist()
    df["priority_rank"] = df["ward_id"].map({w: i + 1 for i, w in enumerate(order)})
    df["equity_flag"] = df["equity_flag"].fillna(False).astype(bool)
    df["slope_fallback"] = df["slope_fallback"].fillna(True).astype(bool)

    df["lst_mean"] = df["lst_mean"].round(2)
    df["ndvi_mean"] = df["ndvi_mean"].round(3)
    df["pop_density"] = df["pop_density"].round(1)
    df["priority_score"] = df["priority_score"].round(2)
    df["top_drivers"] = df["ward_id"].map(lambda w: _top_drivers(drivers, w))
    df["summary"] = df["ward_id"].map(lambda w: summaries.get(w, ""))
    df["slope"] = df["slope_used"].round(3)
    df["slope_r2"] = df["r2"].round(3)
    df["slope_n"] = df["n"].astype(int)

    cols = [
        "ward_id", "ward_name", "lst_mean", "ndvi_mean", "population", "pop_density",
        "plantable_area_m2", "top_drivers", "slope", "slope_r2", "slope_n",
        "slope_fallback", "priority_score", "priority_rank", "equity_flag", "summary",
        "data_source",
    ]
    out = wards[["ward_id", "geometry"]].merge(df[cols], on="ward_id", how="left")
    out = out[out["lst_mean"].notna()].reset_index(drop=True)

    meta = {
        "city": config.CITY,
        "claim": "A spatially validated planting prioritiser that ranks where a tree cools most for the most exposed people.",
        "data_source": str(df["data_source"].iloc[0]),
        "ward_set": "bbmp_243_real (KGIS/DataMeet 243-ward BBMP boundaries)",
        "n_wards": int(len(out)),
        "center": list(config.MAP_CENTER),
        "bbox": config.BBOX,
        "grid_size_m": config.GRID_SIZE_M_SYNTHETIC,
        "summer_window": list(config.SUMMER_WINDOW),
        "overall_r2": metrics["overall_r2"],
        "overall_rmse_c": metrics["overall_rmse_c"],
        "grouping": metrics["grouping"],
        "n_folds": metrics["n_folds"],
        "city_ndvi_slope_c_per_unit": metrics.get("city_ndvi_slope_c_per_unit"),
        "summary_source": "template" if not any(
            c.get("source") == "claude" for c in json.loads(config.SUMMARY_CACHE.read_text()).values()
        ) else "claude",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "footer": config.DATA_LABEL,
        "cooling_label": "Correlational scenario, not a causal or field-measured result.",
    }
    return out, meta


def main() -> None:
    out, meta = build()
    out.to_file(config.WARDS_GEOJSON, driver="GeoJSON")

    hottest = out.sort_values("lst_mean", ascending=False).head(10)
    config.TOP10_HOTTEST_JSON.write_text(
        json.dumps(
            {
                "data_source": meta["data_source"],
                "wards": [
                    {
                        "rank": i + 1,
                        "ward_id": r.ward_id,
                        "ward_name": r.ward_name,
                        "lst_mean": float(r.lst_mean),
                        "ndvi_mean": float(r.ndvi_mean),
                        "population": int(r.population),
                        "pop_density": float(r.pop_density),
                        "equity_flag": bool(r.equity_flag),
                    }
                    for i, r in enumerate(hottest.itertuples())
                ],
            },
            indent=2,
        )
    )
    (config.DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[outputs] wards.geojson: {len(out)} features, "
          f"{config.WARDS_GEOJSON.stat().st_size / 1e6:.2f} MB")
    print(f"[outputs] wards with planting candidates: "
          f"{int((out['priority_score'] > 0).sum())}; equity flagged: "
          f"{int(out['equity_flag'].sum())}")
    print("[outputs] top-5 hottest wards:")
    print(hottest.head(5)[["ward_name", "lst_mean", "ndvi_mean", "equity_flag"]]
          .to_string(index=False))
    print(f"[outputs] wrote top10_hottest.json, meta.json; "
          f"data_source={meta['data_source']}, summaries={meta['summary_source']}")


if __name__ == "__main__":
    sys.exit(main())
