"""XGBoost LST model with SPATIAL block cross-validation + SHAP drivers.

Grouping: GroupKFold over ward_id (5 folds). Whole wards are held out, so no
cell adjacent to a training cell is ever scored as validation - this is the
spatial block CV the brief requires, and random splits are never used.
Trade-off (documented in README): ward groups are unbalanced in area, so folds
differ in size; GroupKFold keeps each ward intact rather than rebalancing.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from pipeline import config
from pipeline.features import load_grid

N_FOLDS = 5
SHAP_PER_WARD = 40  # sampled rows per ward for SHAP (~9.7k rows total)
SHAP_MAX_ROWS = 10_000


def new_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
    )


def spatial_cv(df: pd.DataFrame) -> dict:
    """Pooled out-of-fold predictions with ward-level GroupKFold."""
    X, y = df[config.FEATURES], df["lst"]
    groups = df["ward_id"].astype(str)
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(df), np.nan)

    print(f"[model] GroupKFold by ward_id: {N_FOLDS} folds, {groups.nunique()} groups, {len(df)} cells")
    folds = []
    for i, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
        model = new_model()
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        oof[va] = pred
        r2 = r2_score(y.iloc[va], pred)
        rmse = float(np.sqrt(mean_squared_error(y.iloc[va], pred)))
        folds.append(
            {
                "fold": i,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_wards_val": int(groups.iloc[va].nunique()),
                "r2": round(float(r2), 4),
                "rmse_c": round(rmse, 4),
            }
        )
        print(f"[model] fold {i}: train {len(tr):5d} val {len(va):5d} "
              f"wards {folds[-1]['n_wards_val']:3d}  R2 {r2:.3f}  RMSE {rmse:.2f} C")

    mask = ~np.isnan(oof)
    overall_r2 = float(r2_score(y[mask], oof[mask]))
    overall_rmse = float(np.sqrt(mean_squared_error(y[mask], oof[mask])))
    print(f"[model] pooled out-of-fold: R2 {overall_r2:.3f}  RMSE {overall_rmse:.2f} C "
          f"(n={int(mask.sum())})")
    return {
        "n_folds": N_FOLDS,
        "grouping": "GroupKFold by ward_id (whole wards held out, no random splits)",
        "n_samples": int(len(df)),
        "n_wards": int(groups.nunique()),
        "features": config.FEATURES,
        "folds": folds,
        "overall_r2": round(overall_r2, 4),
        "overall_rmse_c": round(overall_rmse, 4),
    }


def shap_analysis(df: pd.DataFrame, model: XGBRegressor) -> tuple[pd.DataFrame, dict]:
    """Global mean|SHAP| and per-ward top-3 drivers on a bounded sample."""
    import shap

    # Shuffle once, then take up to SHAP_PER_WARD rows per ward (some wards are
    # smaller than the quota), capped at SHAP_MAX_ROWS overall.
    sample = (
        df.sample(frac=1.0, random_state=config.RANDOM_SEED)
        .groupby("ward_id", group_keys=False)
        .head(SHAP_PER_WARD)
        .head(SHAP_MAX_ROWS)
        .reset_index(drop=True)
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    print(f"[model] SHAP sample: {len(sample)} cells across {sample['ward_id'].nunique()} wards")

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(sample[config.FEATURES]))
    if values.ndim == 3:  # some shap versions return (n, features, outputs)
        values = values[..., 0]
    shap_df = pd.DataFrame(values, columns=config.FEATURES)
    shap_df["ward_id"] = sample["ward_id"].astype(str).values
    shap_df["ward_name"] = sample["ward_name"].values

    mean_abs = shap_df[config.FEATURES].abs().mean().sort_values(ascending=False)
    global_df = pd.DataFrame(
        {
            "feature": mean_abs.index,
            "mean_abs_shap_c": mean_abs.values.round(4),
            "share": (mean_abs / mean_abs.sum()).round(4).values,
        }
    )

    wards: dict[str, dict] = {}
    for ward_id, rows in shap_df.groupby("ward_id"):
        abs_mean = rows[config.FEATURES].abs().mean()
        share = abs_mean / abs_mean.sum() if abs_mean.sum() > 0 else abs_mean
        top = []
        for feature in abs_mean.sort_values(ascending=False).index[:3]:
            # Direction: does the feature push LST up or down in this ward?
            direction = _direction(rows[feature].values, sample.loc[rows.index, feature].values)
            top.append(
                {
                    "feature": feature,
                    "direction": direction,
                    "share": round(float(share[feature]), 4),
                    "mean_shap_c": round(float(rows[feature].mean()), 4),
                }
            )
        wards[str(ward_id)] = {"ward_name": str(rows["ward_name"].iloc[0]), "top_drivers": top}

    return global_df, {"wards": wards, "note": "SHAP on final in-sample model; descriptive only."}


def _direction(shap_values: np.ndarray, feature_values: np.ndarray) -> str:
    """Correlate SHAP contribution with the feature value to get its sign."""
    if np.std(feature_values) < 1e-9 or np.std(shap_values) < 1e-9:
        return "neutral"
    corr = float(np.corrcoef(shap_values, feature_values)[0, 1])
    return "hot" if corr > 0 else "cooling"


def main() -> None:
    df, _ = load_grid()
    metrics = spatial_cv(df)
    metrics["data_source"] = str(df["data_source"].iloc[0])

    # Final fit on all cells for SHAP (and for the cooling-efficiency context).
    final = new_model()
    final.fit(df[config.FEATURES], df["lst"])
    global_df, by_ward = shap_analysis(df, final)

    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    global_df.to_csv(config.SHAP_GLOBAL_CSV, index=False)
    config.SHAP_WARD_JSON.write_text(json.dumps(by_ward, indent=1))

    print("[model] global mean|SHAP| (C):")
    print(global_df.to_string(index=False))
    print(f"[model] wrote {config.METRICS_JSON.name}, {config.SHAP_GLOBAL_CSV.name}, "
          f"{config.SHAP_WARD_JSON.name}")
    print(f"[model] example ward drivers: "
          f"{json.dumps(by_ward['wards'][list(by_ward['wards'])[0]]['top_drivers'][:2])}")


if __name__ == "__main__":
    sys.exit(main())
