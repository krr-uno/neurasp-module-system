#!/usr/bin/env python3
"""Run the four MNIST tasks, then all six card variants, repeated N times.

Usage (in your activated neurasp5 environment):
    python scripts/baseline/run_benchmarks.py --runs 5
    python scripts/baseline/run_benchmarks.py --dry-run

Each batch writes runs.csv, result.csv (means/population standard deviations),
and one console log per command. Accuracies are percentages parsed from the
original console output, which rounds them to two decimals. Unreported metrics
stay blank: member and Add2x2 scripts report training time. wall_seconds measures the entire subprocess;
reported_*_seconds retain the original scripts' differing timing definitions.

The runner forwards --device (CPU by default), preserving training settings and caches.
Card variants run in order: sum_2, sum_3, sum_4, prodsum_2, prodsum_3, prodsum_4.
Use neurasp5-gpu with --device gpu for CUDA. Existing card caches may make later runs
faster, just as with manual execution.
Pass --no-cache to recompute stable models for every observation in all tasks,
including card validation and testing, without reading or writing their caches.
Use --epochs and --batch-size to override training settings for every task.
Use --tasks to select tasks, --seed to seed every task, and --checkpoint_freq
to override the card validation/checkpoint interval (0 disables it entirely).
Failed commands are logged, excluded from statistics, and do not stop the batch.
"""

import argparse
import csv
import json
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CARD_VARIANTS = ("sum_2", "sum_3", "sum_4", "prodsum_2", "prodsum_3", "prodsum_4")
# Task name, working directory, Python arguments, whether to set PYTHONPATH=.
COMMANDS = (
    ("mnistAdd", ".", ["examples/baseline/mnistAdd/mnistAdd.py"], True),
    ("member3", "examples/baseline/member3", ["train.py"], False),
    ("member5", "examples/baseline/member5", ["train.py"], False),
    ("add2x2", "examples/baseline/add2x2", ["train.py"], False),
) + tuple(
    (f"card_{variant}", ".", ["examples/baseline/card_arithmetic/train.py",
                             "--image_dir", "./examples/baseline/card_arithmetic/images",
                             "--variant", variant], True)
    for variant in CARD_VARIANTS
)
MEASURES = ("wall_seconds", "reported_training_seconds", "reported_total_seconds",
            "downstream_accuracy_pct", "latent_accuracy_pct")
FIELDS = ("task", "run", "status", "exit_code", "device", "downstream_test_examples", "downstream_protocol", "downstream_test_sha256", *MEASURES, "log_file", "notes")
NUMBER = r"([0-9]+(?:\.[0-9]+)?)"


def parse_metrics(task, output):
    """Extract test metrics only; never substitute validation/train accuracy."""
    output = output.replace("\r", "\n")
    metrics = {key: None for key in MEASURES}
    patterns = {
        "downstream_accuracy_pct": r"^Downstream test accuracy: " + NUMBER + "%",
        "reported_training_seconds": r"--- total time for training: " + NUMBER + r" seconds ---",
        "reported_total_seconds": r"--- total time from beginning: " + NUMBER + r" seconds ---",
    }
    if task == "mnistAdd":
        pairs = re.findall(r"^[ \t]*" + NUMBER + r"\t" + NUMBER + r"[ \t]*$", output, re.MULTILINE)
        if pairs:
            metrics["latent_accuracy_pct"] = float(pairs[-1][1])
    elif task in ("member3", "member5", "add2x2"):
        patterns["latent_accuracy_pct"] = r"^Test Acc: " + NUMBER + "%"
    else:
        patterns["latent_accuracy_pct"] = r"^Latent test accuracy for card: " + NUMBER + "%"
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output, re.MULTILINE)
        if matches:
            metrics[key] = float(matches[-1])
    devices = re.findall(r"^Device: (\S+)", output, re.MULTILINE)
    metrics["device"] = devices[-1] if devices else ""
    details = re.findall(r"^Downstream evaluation: (.+)$", output, re.MULTILINE)
    if details:
        metadata = json.loads(details[-1])
        for key in ("downstream_test_examples", "downstream_protocol", "downstream_test_sha256"):
            metrics[key] = metadata[key]
    return metrics


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(directory, rows):
    write_csv(directory / "runs.csv", rows, FIELDS)
    summaries = []
    for task, *_ in COMMANDS:
        attempts = [row for row in rows if row["task"] == task]
        successful = [row for row in attempts if row["status"] == "ok"]
        summary = {"task": task, "completed_runs": len(attempts),
                   "successful_runs": len(successful), "failed_runs": len(attempts) - len(successful),
                   "std_ddof": 0}
        for key in MEASURES:
            values = [row[key] for row in successful if row[key] is not None]
            summary[key + "_n"] = len(values)
            summary[key + "_mean"] = statistics.mean(values) if values else None
            summary[key + "_std"] = statistics.pstdev(values) if values else None
        summaries.append(summary)
    write_csv(directory / "result.csv", summaries, list(summaries[0]))


def command_for(spec, device, no_cache=False, checkpoint_freq=None, seed=None, epochs=None, batch_size=None):
    command = [sys.executable, *spec[2], "--device", device]
    if no_cache:
        command.append("--no-cache")
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if checkpoint_freq is not None and spec[0].startswith("card_"):
        command.extend(["--checkpoint_freq", str(checkpoint_freq)])
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])
    return command


def run_command(spec, repetition, directory, device="cpu", no_cache=False,
                checkpoint_freq=None, seed=None, epochs=None, batch_size=None):
    task, folder, arguments, set_pythonpath = spec
    environment = os.environ.copy()
    command = command_for(spec, device, no_cache, checkpoint_freq, seed, epochs, batch_size)
    working_directory = ROOT / folder
    if task.startswith("card_"):
        # Card scripts write their JSONL output under results/ relative to cwd.
        # Give them the benchmark directory so their native result file is saved
        # alongside the logs and CSV reports.
        command[1] = str(ROOT / arguments[0])
        image_dir_index = command.index("--image_dir") + 1
        command[image_dir_index] = str(ROOT / command[image_dir_index])
        command.extend(["--results-dir", str(directory)])
        working_directory = directory
    elif set_pythonpath:
        environment["PYTHONPATH"] = "."
    log_path = directory / f"{repetition:03d}_{task}.log"
    print(f"Run {repetition}: {task}; log: {log_path}", flush=True)
    notes = ["Stable-model caching disabled"] if no_cache else []
    if seed is not None:
        notes.append(f"Seed: {seed}")
    if checkpoint_freq is not None and task.startswith("card_"):
        notes.append(f"Checkpoint frequency: {checkpoint_freq}")
    if epochs is not None:
        notes.append(f"Epochs: {epochs}")
    if batch_size is not None:
        notes.append(f"Batch size: {batch_size}")
    started = time.perf_counter()
    with log_path.open("w") as log:
        try:
            result = subprocess.run(command, cwd=working_directory, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
            exit_code = result.returncode
        except OSError as exc:
            exit_code = -1
            notes.append(str(exc))
            log.write(str(exc) + "\n")
    elapsed = time.perf_counter() - started
    metrics = parse_metrics(task, log_path.read_text(errors="replace"))
    metrics["wall_seconds"] = elapsed
    status = "ok" if exit_code == 0 else "failed"
    if status == "ok" and (metrics["latent_accuracy_pct"] is None or
                           metrics["downstream_accuracy_pct"] is None):
        status = "missing_metrics"
        notes.append("Expected test output was not found; inspect the log")
    expected_device = 'cuda' if device == 'gpu' else 'cpu'
    if status == 'ok' and metrics['device'] != expected_device:
        status = 'device_mismatch'
        notes.append(f"Requested {expected_device}, reported {metrics['device']!r}")
    if exit_code:
        notes.append(f"Process exited with code {exit_code}; inspect the log")
    print(f"  {status}; elapsed {elapsed:.2f}s; "
          f"downstream={metrics['downstream_accuracy_pct']}; latent={metrics['latent_accuracy_pct']}", flush=True)
    return {"task": task, "run": repetition, "status": status, "exit_code": exit_code,
            **metrics, "log_file": str(log_path), "notes": "; ".join(notes)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions of all ten tasks (default: 1)")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--no-cache", action="store_true", help="Disable stable-model caching in all examples")
    parser.add_argument("--seed", type=int, help="Use the same training seed for every task and repetition")
    parser.add_argument("--checkpoint_freq", "--checkpoint-freq", type=int,
                        help="Card validation/checkpoint interval; 0 disables it entirely")
    parser.add_argument("--epochs", type=int, help="Training epochs for every task")
    parser.add_argument("--batch_size", "--batch-size", type=int, help="Training tuples per batch for every task")
    parser.add_argument("--tasks", nargs="+", choices=[spec[0] for spec in COMMANDS],
                        help="Run selected tasks; defaults to all ten")
    parser.add_argument("--output-dir", type=Path, help="New directory for CSV reports and logs (default: benchmark_results/baseline_<device>_<timestamp>/)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training")
    args = parser.parse_args(argv)
    commands = [spec for spec in COMMANDS if not args.tasks or spec[0] in args.tasks]
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.checkpoint_freq is not None and args.checkpoint_freq < 0:
        parser.error("--checkpoint_freq must be nonnegative")
    for name in ("epochs", "batch_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name} must be at least 1")
    directory = (args.output_dir or ROOT / "benchmark_results" /
                 (f"baseline_{args.device}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if args.dry_run:
        print(f"Results would be written to: {directory}")
        print(f"Repeat this sequence {args.runs} time(s):")
        for spec in commands:
            _, folder, arguments, set_pythonpath = spec
            prefix = "PYTHONPATH=. " if set_pythonpath else ""
            print(f"  (cd {shlex.quote(str(ROOT / folder))} && "
                  f"{prefix}{shlex.join(command_for(spec, args.device, args.no_cache, args.checkpoint_freq, args.seed, args.epochs, args.batch_size))})")
        return 0
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists; choose a new one: {directory}")
    print(f"Results: {directory}\nUnreported metrics remain blank. Standard deviation uses ddof=0.", flush=True)
    rows = []
    save_results(directory, rows)
    for repetition in range(1, args.runs + 1):
        for spec in commands:
            row = run_command(spec, repetition, directory, args.device, args.no_cache,
                              args.checkpoint_freq, args.seed, args.epochs, args.batch_size)
            rows.append(row)
            result_path = directory / f"{row['task']}_run_metrics.jsonl"
            with result_path.open("a") as result_file:
                result_file.write(json.dumps(row) + "\n")
            save_results(directory, rows)
    print(f"Per-run results: {directory / 'runs.csv'}\nMeans and standard deviations: {directory / 'result.csv'}")
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
