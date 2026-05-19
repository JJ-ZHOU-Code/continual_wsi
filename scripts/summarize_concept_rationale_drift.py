#!/usr/bin/env python3
"""Aggregate concept-rationale drift smoke-test results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data_2_4T/data_zjj/continual_wsi/concept_rationale_drift/relevance_5seed")
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for model, metrics in payload["models"].items():
            row: dict[str, object] = {
                "seed": payload["seed"],
                "model": model,
                "path": str(path),
            }
            config = payload.get("config", {})
            for key in ("l2_lambda", "anti_penalty", "anti_threshold", "subspace_lambda", "score_power", "relevance_power"):
                if key in config:
                    row[key] = float(config[key])
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = float(value)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No result.json files found under {root}/*/")

    metric_keys = [
        "old_corr_acc",
        "reversed_acc",
        "neutral_random_mean_acc",
        "crc",
        "cdr",
        "subspace_rotation_deg",
        "scr",
        "intervention_agreement",
        "old_minus_reversed_acc",
    ]
    aggregate: list[dict[str, object]] = []
    group_keys = ("model", "l2_lambda", "anti_penalty", "anti_threshold", "subspace_lambda")
    groups = sorted({tuple(row.get(key, "") for key in group_keys) for row in rows})
    for group_id in groups:
        group = [row for row in rows if tuple(row.get(key, "") for key in group_keys) == group_id]
        out: dict[str, object] = {"n": len(group)}
        for key, value in zip(group_keys, group_id):
            out[key] = value
        for key in metric_keys:
            values = [float(row[key]) for row in group if key in row]
            out[f"{key}_mean"] = stats.mean(values)
            out[f"{key}_std"] = stats.pstdev(values) if len(values) > 1 else 0.0
        aggregate.append(out)

    summary_rows = root / "summary_rows.csv"
    with summary_rows.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_csv = root / "aggregate.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    report = {
        "rows": len(rows),
        "summary_rows": str(summary_rows),
        "aggregate_csv": str(aggregate_csv),
        "aggregate": aggregate,
    }
    (root / "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
