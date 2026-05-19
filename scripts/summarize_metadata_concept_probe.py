#!/usr/bin/env python3
"""Aggregate metadata concept-probe drift sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from pathlib import Path


METRICS = [
    "test_balanced_acc",
    "worst_group_acc",
    "test_balanced_env_pred_corr",
    "old_minus_reversed_acc",
    "crc",
    "cdr",
    "scr",
    "subspace_rotation_deg",
    "intervention_agreement",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        for model, metrics in payload["models"].items():
            row: dict[str, object] = {
                "seed": payload["seed"],
                "model": model,
                "l2_lambda": cfg["l2_lambda"],
                "anti_penalty": cfg["anti_penalty"],
                "anti_threshold": cfg["anti_threshold"],
                "subspace_lambda": cfg["subspace_lambda"],
                "path": str(path),
            }
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = float(value)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No result files under {root}/*/result.json")

    groups = sorted({(r["model"], r["l2_lambda"], r["anti_penalty"]) for r in rows})
    agg: list[dict[str, object]] = []
    for model, l2, pen in groups:
        group = [r for r in rows if (r["model"], r["l2_lambda"], r["anti_penalty"]) == (model, l2, pen)]
        out: dict[str, object] = {"model": model, "l2_lambda": l2, "anti_penalty": pen, "n": len(group)}
        for key in METRICS:
            vals = [float(r[key]) for r in group if key in r]
            out[f"{key}_mean"] = stats.mean(vals)
            out[f"{key}_std"] = stats.pstdev(vals) if len(vals) > 1 else 0.0
        out["abs_env_corr_mean"] = abs(float(out["test_balanced_env_pred_corr_mean"]))
        out["pareto_score"] = (
            float(out["test_balanced_acc_mean"])
            + 0.5 * float(out["worst_group_acc_mean"])
            - 0.2 * float(out["scr_mean"])
            - 0.01 * float(out["subspace_rotation_deg_mean"])
            - 0.1 * float(out["abs_env_corr_mean"])
        )
        agg.append(out)

    with (root / "summary_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        writer.writeheader()
        writer.writerows(agg)

    top = sorted([r for r in agg if r["model"] == "metadata_cca"], key=lambda r: float(r["pareto_score"]), reverse=True)[:10]
    baselines = []
    seen = set()
    for r in agg:
        if r["model"] not in {"finetune", "l2_all", "naive_probe_l2", "random_score_l2"}:
            continue
        key = (r["model"], r["l2_lambda"])
        if key in seen:
            continue
        seen.add(key)
        baselines.append(r)
    report = {
        "rows": len(rows),
        "aggregate_csv": str(root / "aggregate.csv"),
        "top_metadata_cca": top,
        "baselines": sorted(baselines, key=lambda r: (str(r["model"]), float(r["l2_lambda"]))),
    }
    (root / "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
