#!/usr/bin/env python3
"""Launch a small parameter sweep for shortcut-reversal diagnostics.

The runner intentionally stays dependency-light so it can run inside the
existing clam environment. Each cell calls shortcut_reversal_smoke.py and writes
an isolated result directory. Existing completed result.json files are skipped.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    seed: int
    strength: float
    l2_lambda: float
    shortcut_penalty: float

    @property
    def job_id(self) -> str:
        return (
            f"seed{self.seed}_str{self.strength:g}_"
            f"l2{self.l2_lambda:g}_pen{self.shortcut_penalty:g}"
        ).replace(".", "p")


def parse_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def metric(payload: dict, model: str, key: str) -> float:
    return float(payload["models"][model][key])


def shortcut_gap(payload: dict, model: str) -> float:
    return metric(payload, model, "test_old_corr_acc") - metric(payload, model, "test_reversed_acc")


def robustness(payload: dict, model: str) -> float:
    return 0.5 * (
        metric(payload, model, "test_neutral_acc")
        + metric(payload, model, "test_random_acc")
    )


def summarize_one(result_path: Path) -> dict[str, object]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    row: dict[str, object] = {
        "result_path": str(result_path),
        "seed": payload["seed"],
        "strength": payload["shortcut_strength"],
        "noise": payload["shortcut_noise"],
        "l2_lambda": payload["l2_lambda"],
        "shortcut_penalty": payload["shortcut_penalty"],
    }
    for model in ["task1_only", "finetune", "l2_all", "selective_l2", "csr_aug"]:
        for key in [
            "env1_train_acc",
            "env2_train_acc",
            "test_old_corr_acc",
            "test_reversed_acc",
            "test_neutral_acc",
            "test_random_acc",
        ]:
            row[f"{model}_{key}"] = metric(payload, model, key)
        row[f"{model}_old_minus_reversed"] = shortcut_gap(payload, model)
        row[f"{model}_neutral_random_mean"] = robustness(payload, model)
    row["csr_aug_minus_l2_all_robustness"] = (
        row["csr_aug_neutral_random_mean"] - row["l2_all_neutral_random_mean"]
    )
    row["csr_aug_minus_finetune_robustness"] = (
        row["csr_aug_neutral_random_mean"] - row["finetune_neutral_random_mean"]
    )
    return row


def write_summary(out_root: Path) -> None:
    rows = [summarize_one(path) for path in sorted(out_root.glob("*/result.json"))]
    if not rows:
        return

    summary_csv = out_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    agg: dict[str, object] = {"num_completed": len(rows), "models": {}}
    for model in ["task1_only", "finetune", "l2_all", "selective_l2", "csr_aug"]:
        model_stats: dict[str, float] = {}
        for key in [
            "test_old_corr_acc",
            "test_reversed_acc",
            "test_neutral_acc",
            "test_random_acc",
            "old_minus_reversed",
            "neutral_random_mean",
        ]:
            values = [float(row[f"{model}_{key}"]) for row in rows]
            model_stats[f"{key}_mean"] = sum(values) / len(values)
        agg["models"][model] = model_stats
    for key in ["csr_aug_minus_l2_all_robustness", "csr_aug_minus_finetune_robustness"]:
        values = [float(row[key]) for row in rows]
        agg[f"{key}_mean"] = sum(values) / len(values)

    (out_root / "aggregate.json").write_text(
        json.dumps(agg, indent=2),
        encoding="utf-8",
    )


def run_job(
    job: Job,
    *,
    script: Path,
    out_root: Path,
    cache: str,
    noise: float,
    epochs_task1: int,
    epochs_task2: int,
    lr: float,
    force: bool,
) -> tuple[str, str]:
    out_dir = out_root / job.job_id
    result_path = out_dir / "result.json"
    log_path = out_dir / "run.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    if result_path.exists() and not force:
        return job.job_id, "skipped"

    cmd = [
        sys.executable,
        str(script),
        "--cache",
        cache,
        "--out-dir",
        str(out_dir),
        "--seed",
        str(job.seed),
        "--strength",
        str(job.strength),
        "--noise",
        str(noise),
        "--epochs-task1",
        str(epochs_task1),
        "--epochs-task2",
        str(epochs_task2),
        "--lr",
        str(lr),
        "--l2-lambda",
        str(job.l2_lambda),
        "--shortcut-penalty",
        str(job.shortcut_penalty),
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        return job.job_id, f"failed:{proc.returncode}"
    elapsed = time.time() - started
    return job.job_id, f"completed:{elapsed:.1f}s"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="scripts/shortcut_reversal_smoke.py")
    parser.add_argument("--cache", default="/data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/mean_features_max60_seed7.pt")
    parser.add_argument("--out-root", default="/data_2_4T/data_zjj/continual_wsi/shortcut_sweeps/latest")
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--strengths", default="2,4,6,8")
    parser.add_argument("--l2-lambdas", default="20,80")
    parser.add_argument("--shortcut-penalties", default="0.0,0.1")
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--epochs-task1", type=int, default=300)
    parser.add_argument("--epochs-task2", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    script = Path(args.script)

    jobs = [
        Job(seed=seed, strength=strength, l2_lambda=l2_lambda, shortcut_penalty=penalty)
        for seed, strength, l2_lambda, penalty in itertools.product(
            parse_ints(args.seeds),
            parse_floats(args.strengths),
            parse_floats(args.l2_lambdas),
            parse_floats(args.shortcut_penalties),
        )
    ]

    manifest = {
        "num_jobs": len(jobs),
        "seeds": parse_ints(args.seeds),
        "strengths": parse_floats(args.strengths),
        "l2_lambdas": parse_floats(args.l2_lambdas),
        "shortcut_penalties": parse_floats(args.shortcut_penalties),
        "noise": args.noise,
        "epochs_task1": args.epochs_task1,
        "epochs_task2": args.epochs_task2,
        "lr": args.lr,
        "max_workers": args.max_workers,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Launching {len(jobs)} jobs into {out_root}", flush=True)
    statuses: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(
                run_job,
                job,
                script=script,
                out_root=out_root,
                cache=args.cache,
                noise=args.noise,
                epochs_task1=args.epochs_task1,
                epochs_task2=args.epochs_task2,
                lr=args.lr,
                force=args.force,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            job_id, status = future.result()
            statuses[job_id] = status
            (out_root / "status.json").write_text(
                json.dumps(statuses, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"{job_id}: {status}", flush=True)
            write_summary(out_root)

    write_summary(out_root)
    failed = {k: v for k, v in statuses.items() if v.startswith("failed")}
    if failed:
        print(json.dumps({"failed": failed}, indent=2), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

