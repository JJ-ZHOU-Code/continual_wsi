#!/usr/bin/env python3
"""Summarize feature-cluster proxy shift sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


def mean(rows: list[dict[str, float]], key: str) -> float:
    return float(stats.mean(float(row[key]) for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data_2_4T/data_zjj/continual_wsi/feature_cluster_shift")
    parser.add_argument("--out", default="/data_2_4T/data_zjj/continual_wsi/feature_cluster_shift/sweep_summary")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("sweep_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for model, metrics in payload["models"].items():
            row: dict[str, object] = {
                "result_path": str(path),
                "seed": payload["seed"],
                "l2_lambda": payload["l2_lambda"],
                "anti_penalty": payload["anti_penalty"],
                "anti_threshold": payload["anti_threshold"],
                "score_power": payload["score_power"],
                "model": model,
                "stability_raw_mean": payload["stability_summary"]["raw_mean"],
                "stability_raw_min": payload["stability_summary"]["raw_min"],
                "stability_powered_mean": payload["stability_summary"]["powered_mean"],
            }
            for key, value in metrics.items():
                row[key] = value
            rows.append(row)

    if not rows:
        raise SystemExit(f"No result files found under {root}/sweep_*/result.json")

    summary_csv = out / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_cfg_model: dict[tuple[float, float, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        cfg = (
            float(row["l2_lambda"]),
            float(row["anti_penalty"]),
            float(row["anti_threshold"]),
            str(row["model"]),
        )
        by_cfg_model[cfg].append(row)

    agg_rows: list[dict[str, object]] = []
    metric_keys = [
        "test_balanced_acc",
        "test_old_corr_acc",
        "test_reversed_acc",
        "worst_group_acc",
        "test_balanced_env_pred_corr",
        "old_minus_reversed_acc",
        "weight_norm",
    ]
    for (l2_lambda, anti_penalty, anti_threshold, model), group in sorted(by_cfg_model.items()):
        agg: dict[str, object] = {
            "l2_lambda": l2_lambda,
            "anti_penalty": anti_penalty,
            "anti_threshold": anti_threshold,
            "model": model,
            "num_seeds": len(group),
        }
        for key in metric_keys:
            agg[f"{key}_mean"] = mean(group, key)
        agg["abs_env_corr_mean"] = abs(float(agg["test_balanced_env_pred_corr_mean"]))
        agg["abs_old_minus_reversed_mean"] = abs(float(agg["old_minus_reversed_acc_mean"]))
        agg["pareto_score"] = (
            float(agg["test_balanced_acc_mean"])
            + 0.5 * float(agg["worst_group_acc_mean"])
            - 0.2 * float(agg["abs_env_corr_mean"])
            - 0.2 * float(agg["abs_old_minus_reversed_mean"])
        )
        agg_rows.append(agg)

    aggregate_csv = out / "aggregate.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    top_streaming = sorted(
        [row for row in agg_rows if row["model"] == "streaming_score_anti"],
        key=lambda row: float(row["pareto_score"]),
        reverse=True,
    )[:12]
    top_balanced = sorted(
        [row for row in agg_rows if row["model"] == "streaming_score_anti"],
        key=lambda row: float(row["test_balanced_acc_mean"]),
        reverse=True,
    )[:12]
    baselines = [
        row
        for row in agg_rows
        if row["l2_lambda"] == 10.0
        and row["anti_penalty"] == 0.0
        and row["anti_threshold"] == 0.1
        and row["model"] in {"finetune", "l2_all", "random_score_l2"}
    ]
    report = {
        "num_raw_rows": len(rows),
        "num_aggregate_rows": len(agg_rows),
        "summary_csv": str(summary_csv),
        "aggregate_csv": str(aggregate_csv),
        "top_streaming_pareto": top_streaming,
        "top_streaming_balanced_acc": top_balanced,
        "reference_baselines_l2_10": baselines,
    }
    (out / "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

