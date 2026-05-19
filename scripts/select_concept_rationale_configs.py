#!/usr/bin/env python3
"""Print concise Pareto summaries for concept-rationale drift sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.aggregate).open("r", encoding="utf-8")))
    for row in rows:
        row["pareto_score"] = str(
            f(row, "neutral_random_mean_acc_mean")
            - 0.2 * f(row, "scr_mean")
            - 0.01 * f(row, "subspace_rotation_deg_mean")
            - 0.1 * abs(f(row, "old_minus_reversed_acc_mean"))
        )

    cols = [
        "model",
        "l2_lambda",
        "anti_penalty",
        "neutral_random_mean_acc_mean",
        "old_corr_acc_mean",
        "reversed_acc_mean",
        "scr_mean",
        "subspace_rotation_deg_mean",
        "intervention_agreement_mean",
        "old_minus_reversed_acc_mean",
        "pareto_score",
    ]

    def print_table(title: str, table: list[dict[str, str]]) -> None:
        print(title)
        print(",".join(cols))
        for row in table:
            print(",".join(row.get(c, "") for c in cols))

    cca = [row for row in rows if row["model"] == "cca_anti_subspace"]
    cca = sorted(cca, key=lambda row: f(row, "pareto_score"), reverse=True)[: args.top]
    print_table("TOP_CCA", cca)

    seen = set()
    baselines = []
    for row in rows:
        if row["model"] not in {"finetune", "l2_all", "naive_concept_l2", "random_score_l2"}:
            continue
        key = (row["model"], row["l2_lambda"])
        if key in seen:
            continue
        seen.add(key)
        baselines.append(row)
    baselines = sorted(baselines, key=lambda row: (row["model"], f(row, "l2_lambda")))
    print()
    print_table("BASELINES", baselines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
