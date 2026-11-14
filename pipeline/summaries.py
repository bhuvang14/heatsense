"""Plain-language 2-3 sentence summary per ward.

If ANTHROPIC_API_KEY is set, summaries are generated at BUILD TIME with the
Claude API (batched, facts-only prompt). Otherwise, or if a call fails, a
deterministic template is used. Results are cached in data/summary_cache.json
keyed by a hash of the ward's facts, so reruns do not re-call the API.

Model name: HEATSENSE_MODEL env var, default below.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pandas as pd

from pipeline import config
from pipeline.cooling import LABEL

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
BATCH_SIZE = 20
PROMPT_RULES = (
    "Rules: exactly 2-3 sentences. Use only the numbers given for that ward; invent nothing. "
    "Say 'surface heat' or 'surface temperature', never 'air temperature'. Do not claim "
    "causation; the cooling number is a correlational scenario. If slope_fallback is true, "
    "mention that a city-wide slope was used because the ward sample was small or noisy. "
    "Return ONLY a JSON array of objects {\"ward_id\": \"...\", \"summary\": \"...\"}."
)


def facts_for_ward(ward: pd.Series, cool: pd.Series, drivers: list[dict]) -> dict:
    """Only the facts a summary is allowed to use."""
    return {
        "ward_id": str(ward["ward_id"]),
        "ward_name": str(ward["ward_name"]),
        "lst_mean_c": round(float(ward["lst_mean"]), 2),
        "ndvi_mean": round(float(ward["ndvi_mean"]), 3),
        "population": int(ward["population"]),
        "pop_density_per_km2": int(round(float(ward["pop_density"]))),
        "top_driver": drivers[0]["feature"] if drivers else "none",
        "second_driver": drivers[1]["feature"] if len(drivers) > 1 else "none",
        "driver_direction": drivers[0]["direction"] if drivers else "neutral",
        "slope_c_per_unit_ndvi": round(float(cool["slope_used"]), 3),
        "slope_r2": round(float(cool["r2"]), 3),
        "slope_fallback": bool(cool["slope_fallback"]),
        "cooling_0_1_ndvi_c": round(-float(cool["slope_used"]) * 0.1, 2),
        "equity_flag": bool(ward.get("equity_flag", False)),
    }


def template_summary(f: dict) -> str:
    """Deterministic fallback (max 3 sentences) using the same facts as the prompt."""
    fb = (
        " (city-wide slope used: this ward's own sample was too small or too noisy)"
        if f["slope_fallback"]
        else ""
    )
    equity = (
        ", and is flagged for heat-equity priority: hot, dense and low in green cover"
        if f["equity_flag"]
        else ""
    )
    return (
        f"{f['ward_name']} averages {f['lst_mean_c']:.1f} C of summer surface heat, with mean "
        f"NDVI {f['ndvi_mean']:.2f} and about {f['population']:,} residents{equity}. "
        f"Its strongest modelled heat driver is {f['top_driver'].replace('_', ' ')} "
        f"({f['driver_direction']}), followed by {f['second_driver'].replace('_', ' ')}. "
        f"A +0.1 NDVI greening scenario implies about {f['cooling_0_1_ndvi_c']:.1f} C less "
        f"surface heat here{fb}."
    )


def _cache_key(f: dict, model: str) -> str:
    blob = json.dumps({"f": f, "m": model, "v": 1}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def _claude_batch(facts: list[dict], model: str) -> dict[str, str]:
    """One batched Claude call. Raises on any failure (caller falls back)."""
    from anthropic import Anthropic  # optional dependency

    client = Anthropic()
    payload = json.dumps(facts, indent=1)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=(
            "You write short, factual ward briefings for a heat-mapping tool. "
            + PROMPT_RULES
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Write one summary per ward from these fact objects:\n" + payload
                ),
            }
        ],
    )
    text = "".join(block.text for block in message.content if block.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    rows = json.loads(text)
    out: dict[str, str] = {}
    for row in rows:
        wid = str(row["ward_id"])
        summary = str(row["summary"]).strip()
        if summary:
            out[wid] = summary
    return out


def generate() -> dict[str, str]:
    wards = pd.read_csv(config.WARD_AGG_CSV, dtype={"ward_id": str})
    cooling = pd.read_csv(config.COOLING_CSV, dtype={"ward_id": str})
    drivers = json.loads(config.SHAP_WARD_JSON.read_text())["wards"]
    priority = pd.read_csv(config.DATA_DIR / "ward_priority.csv", dtype={"ward_id": str})
    from pipeline.priority import equity_flags

    wards = wards.merge(priority[["ward_id", "priority_score", "priority_rank"]], on="ward_id", how="left")
    wards = wards.merge(equity_flags(wards), on="ward_id", how="left")
    wards["equity_flag"] = wards["equity_flag"].fillna(False).astype(bool)

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    model = os.environ.get("HEATSENSE_MODEL", DEFAULT_MODEL)
    cache = json.loads(config.SUMMARY_CACHE.read_text()) if config.SUMMARY_CACHE.exists() else {}

    facts: list[dict] = []
    for _, ward in wards.iterrows():
        cool = cooling[cooling["ward_id"] == ward["ward_id"]]
        if cool.empty:
            continue
        ward_drivers = drivers.get(str(ward["ward_id"]), {}).get("top_drivers", [])
        facts.append(facts_for_ward(ward, cool.iloc[0], ward_drivers))

    results: dict[str, str] = {}
    pending: list[dict] = []
    n_cached = 0
    for f in facts:
        entry = cache.get(f["ward_id"])
        if entry and entry.get("key") == _cache_key(f, model):
            results[f["ward_id"]] = entry["summary"]
            n_cached += 1
        else:
            pending.append(f)

    claude_ids: set[str] = set()
    if key and pending:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            print("[summaries] ANTHROPIC_API_KEY set but `anthropic` not installed -> templates")
            key = ""
        while key and pending:
            batch, pending = pending[:BATCH_SIZE], pending[BATCH_SIZE:]
            try:
                batch_out = _claude_batch(batch, model)
                for f in batch:
                    summary = batch_out.get(f["ward_id"])
                    if summary:
                        results[f["ward_id"]] = summary
                        claude_ids.add(f["ward_id"])
                    else:
                        results[f["ward_id"]] = template_summary(f)
            except Exception as exc:  # noqa: BLE001 - any API issue -> template
                print(f"[summaries] Claude call failed ({exc}); using templates for this batch")
                for f in batch:
                    results.setdefault(f["ward_id"], template_summary(f))

    for f in pending:
        results.setdefault(f["ward_id"], template_summary(f))

    for f in facts:
        cache[f["ward_id"]] = {
            "key": _cache_key(f, model),
            "summary": results[f["ward_id"]],
            "source": "claude" if f["ward_id"] in claude_ids else "template",
        }
    config.SUMMARY_CACHE.write_text(json.dumps(cache, indent=1))

    n_template = len(results) - n_cached - len(claude_ids)
    print(f"[summaries] {len(results)} wards: {len(claude_ids)} from Claude, "
          f"{n_cached} from cache, {n_template} from templates "
          f"(model={model if key else 'none'})")
    return results


def main() -> None:
    summaries = generate()
    (config.DATA_DIR / "summaries.json").write_text(json.dumps(summaries, indent=1))
    sample = next(iter(summaries.items()))
    print(f"[summaries] example ({sample[0]}): {sample[1]}")
    print(f"[summaries] wrote summaries.json ({len(summaries)} wards) and summary_cache.json")
    print(f"[summaries] label: {LABEL}")


if __name__ == "__main__":
    sys.exit(main())
