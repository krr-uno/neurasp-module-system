#!/usr/bin/env python3
"""Run native pipeline examples sequentially, with one results folder per invocation.

Example (from the repository root, in neurasp-gpu):
  python scripts/run_pipeline_benchmarks.py --device gpu --batch-size 16 \
    --tasks mnistAdd card_sum_2 card_sum_3 --no-models-cache

Each task writes runs/NNN_task/{metrics.json,runs.csv}; stdout/stderr go to
logs/NNN_task.log. Aggregate runs.csv/json and result.csv/json are refreshed after
each task. Repetition seeds are seed-base, seed-base+1, ...; standard deviation
uses ddof=0 and includes successful runs only. This branch has MNIST addition,
membership, Add2x2 and six card tasks; it has no model-checkpoint writer.

Build caches before benchmarking by running the same task selection first with
--compile-only (optionally --rebuild-models to force rebuilding), then without it.
Use a different output directory for each invocation. Compile-only runs record
exit status and process timing but produce no training metrics or accuracies.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
CARD_TASKS = tuple(
    f"card_{operation}_{count}"
    for operation in ("sum", "prodsum")
    for count in (2, 3, 4)
)
TASKS = ("mnistAdd", "member3", "member5", "add2x2", *CARD_TASKS)
MODULES = {
    "mnistAdd": "examples.mnistAdd.mnistAdd",
    "member3": "examples.members.member3",
    "member5": "examples.members.member5",
    "add2x2": "examples.add2x2.train",
    **{task: "examples.card_arithmetic.card_arithmetic" for task in CARD_TASKS},
}
MEASURES = (
    "wall_seconds",
    "calendar_seconds",
    "script_seconds",
    "compilation_seconds",
    "training_seconds",
    "evaluation_seconds",
    "downstream_accuracy_pct",
    "latent_accuracy_pct",
)
SETTINGS = (
    "device",
    "precision",
    "deterministic",
    "models_source",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "train_examples",
    "test_examples",
)
FIELDS = (
    "task",
    "run",
    "seed",
    "status",
    "exit_code",
    *SETTINGS,
    *MEASURES,
    "log_file",
    "metrics_file",
    "notes",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse runner options and the native example options to forward."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed-base", "--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--epochs", type=int, help="Override all tasks; defaults: MNIST addition 1, members/Add2x2 3, cards 10"
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        help="Override all tasks; defaults: MNIST .001, cards .0001",
    )
    parser.add_argument("--precision", choices=("ieee", "tf32", "bf16"), default="ieee")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--compile-processes", type=int, default=1)
    parser.add_argument("--clingo-threads", type=int, default=1)
    parser.add_argument(
        "--clingo-parallel-mode", choices=("compete", "split"), default="compete"
    )
    parser.add_argument(
        "--clingo-enum-mode", choices=("auto", "bt", "record"), default="auto"
    )
    parser.add_argument(
        "--models-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse/write compiled models by default; --no-models-cache recompiles each run",
    )
    parser.add_argument(
        "--no-cache", dest="models_cache", action="store_false",
        help="Alias for --no-models-cache: do not read or write compiled-model caches",
    )
    parser.add_argument("--rebuild-models", action="store_true")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile selected tasks without training; caches are enabled by default",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Card images; defaults to the example's images directory",
    )
    parser.add_argument("--preload-images", action="store_true", help="Card tasks only")
    parser.add_argument(
        "--output-dir", type=Path, help="New parent directory for all selected runs (default: benchmark_results/pipeline_<settings>_<timestamp>/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands; create no files or processes",
    )
    args = parser.parse_args(argv)
    if args.runs < 1 or args.seed_base < 0 or args.seed_base + args.runs > 2**32:
        parser.error("runs must be positive; every seed must be in 0..2**32-1")
    if args.batch_size < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("batch size and epochs must be positive")
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("learning rate must be finite and positive")
    if (
        args.workers < 0
        or args.compile_processes < 1
        or not 1 <= args.clingo_threads <= 64
    ):
        parser.error(
            "workers must be nonnegative, compile-processes positive, clingo-threads in 1..64"
        )
    if args.device == "cpu" and args.precision != "ieee":
        parser.error("tf32/bf16 precision requires --device gpu or cuda")
    args.requested_device = args.device
    args.device = "cuda" if args.device == "gpu" else args.device
    args.tasks = list(dict.fromkeys(args.tasks))
    if args.image_dir is not None:
        args.image_dir = args.image_dir.resolve()
    return args


def command_for(
    task: str, seed: int, output: Path, args: argparse.Namespace
) -> list[str]:
    """Construct one native command using this runner's Python environment."""
    command = [
        sys.executable,
        "-B",
        "-u",
        "-m",
        MODULES[task],
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
    ]
    for option in (
        "device",
        "batch_size",
        "precision",
        "workers",
        "compile_processes",
        "clingo_threads",
        "clingo_parallel_mode",
        "clingo_enum_mode",
        "epochs",
        "learning_rate",
    ):
        value = getattr(args, option)
        if value is not None:
            command += ["--" + option.replace("_", "-"), str(value)]
    command.append("--models-cache" if args.models_cache else "--no-models-cache")
    for option in ("deterministic", "rebuild_models", "compile_only"):
        if getattr(args, option):
            command.append("--" + option.replace("_", "-"))
    if task in CARD_TASKS:
        command += ["--variant", task.removeprefix("card_")]
        if args.image_dir is not None:
            command += ["--image-dir", str(args.image_dir)]
        if args.preload_images:
            command.append("--preload-images")
    return command


def utc_now() -> str:
    """Return an unambiguous timestamp for the manifest."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    """Replace a JSON report atomically, preserving its previous version on error."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n"
    )
    temporary.replace(path)


def save_reports(directory: Path, rows: list[dict[str, Any]], tasks: list[str]) -> None:
    """Write per-run reports and per-task means/population standard deviations."""
    summaries = []
    for task in tasks:
        attempted = [row for row in rows if row["task"] == task]
        successful = [row for row in attempted if row["status"] == "ok"]
        summary = {
            "task": task,
            "attempted_runs": len(attempted),
            "successful_runs": len(successful),
            "failed_runs": sum(row["status"] == "failed" for row in attempted),
            "interrupted_runs": sum(
                row["status"] == "interrupted" for row in attempted
            ),
            "std_ddof": 0,
        }
        for key in MEASURES:
            values = [row[key] for row in successful if row.get(key) is not None]
            summary[key + "_n"] = len(values)
            summary[key + "_mean"] = statistics.mean(values) if values else None
            summary[key + "_std"] = statistics.pstdev(values) if values else None
        summaries.append(summary)
    for name, entries, fields in (
        ("runs", rows, FIELDS),
        ("result", summaries, list(summaries[0])),
    ):
        path = directory / f"{name}.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(entries)
        temporary.replace(path)
        write_json(directory / f"{name}.json", entries)


def read_metrics(
    path: Path, task: str, seed: int, args: argparse.Namespace
) -> dict[str, Any]:
    """Reject stale, failed, smoke-test, mismatched, or nonfinite measurements."""
    metrics = json.loads(path.read_text())
    if (
        not isinstance(metrics, dict)
        or metrics.get("task") != task
        or metrics.get("seed") != seed
    ):
        raise ValueError("metrics task/seed does not match the requested run")
    if metrics.get("status") != "ok" or metrics.get("smoke_test"):
        raise ValueError("metrics are not from a successful full run")
    for field in ("device", "precision", "batch_size", "deterministic"):
        if metrics.get(field) != getattr(args, field):
            raise ValueError(f"metrics {field} does not match the requested run")
    for field in ("epochs", "learning_rate"):
        if getattr(args, field) is not None and metrics.get(field) != getattr(
            args, field
        ):
            raise ValueError(f"metrics {field} does not match the requested run")
    for key in (
        "wall_seconds",
        "compilation_seconds",
        "training_seconds",
        "evaluation_seconds",
        "latent_accuracy_pct",
        "downstream_accuracy_pct",
    ):
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"invalid or missing {key}")
        if key.endswith("_pct") and value > 100:
            raise ValueError(f"{key} exceeds 100")
    return metrics


def stop_child(process: subprocess.Popen[Any]) -> None:
    """Stop and reap only the task's process group, including compile/data workers."""
    if process.poll() is not None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif sig == signal.SIGKILL:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def run_one(
    task: str, repetition: int, seed: int, directory: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Run one child to completion, capture its log, and read its final metrics."""
    name = f"{repetition:03d}_{task}"
    # Native scripts reject existing output directories; create only the parent.
    output = directory / "runs" / name
    log_path = directory / "logs" / f"{name}.log"
    row = dict.fromkeys(FIELDS)
    row.update(
        task=task,
        run=repetition,
        seed=seed,
        status="failed",
        notes="",
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        deterministic=args.deterministic,
        log_file=str(log_path),
        metrics_file=None if args.compile_only else str(output / "metrics.json"),
    )
    command = command_for(task, seed, output, args)
    print(f"Run {repetition}: {task}, seed {seed}; log: {log_path}", flush=True)
    start, calendar_start = time.perf_counter(), time.time()
    process = None
    with log_path.open("w") as log:
        log.write("Command: " + shlex.join(command) + "\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            row["exit_code"] = process.wait()
            row["status"] = "ok" if row["exit_code"] == 0 else "failed"
            if row["status"] == "failed":
                row["notes"] = (
                    f'Process exited with code {row["exit_code"]}; inspect the log.'
                )
        except KeyboardInterrupt:
            if process is not None:
                stop_child(process)
                row["exit_code"] = process.returncode
            row.update(
                status="interrupted", notes="Interrupted; no final accuracy reported."
            )
        except OSError as exc:
            if process is not None:
                stop_child(process)
            row.update(exit_code=-1, notes=str(exc))
            log.write(str(exc) + "\n")
    row["wall_seconds"] = time.perf_counter() - start
    row["calendar_seconds"] = time.time() - calendar_start
    if row["status"] == "ok" and args.compile_only:
        row["notes"] = "Compile-only completed; no training or evaluation performed."
    elif row["status"] == "ok":
        try:
            metrics = read_metrics(output / "metrics.json", task, seed, args)
            for key in (
                *SETTINGS,
                "compilation_seconds",
                "training_seconds",
                "evaluation_seconds",
                "downstream_accuracy_pct",
                "latent_accuracy_pct",
            ):
                row[key] = metrics.get(key)
            row["script_seconds"] = metrics["wall_seconds"]
            row["notes"] = str(metrics.get("notes", ""))
            # Child run=1 is local to its directory; retain the repetition above.
        except (OSError, ValueError, TypeError) as exc:
            row.update(status="failed", notes=f"Invalid metrics: {exc}")
    outcome = (
        "compile-only"
        if args.compile_only
        else f"downstream={row['downstream_accuracy_pct']}; latent={row['latent_accuracy_pct']}"
    )
    print(
        f"  {row['status']}; {row['wall_seconds']:.2f}s; "
        f"{outcome}",
        flush=True,
    )
    return row


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected tasks in order; flush reports after each, including interruptions."""
    args = parse_args(argv)
    device_label = "gpu" if args.device == "cuda" else "cpu"
    directory = (
        args.output_dir
        or ROOT
        / "benchmark_results"
        / (
            f"pipeline_{device_label}_batch{args.batch_size}_"
            + ("compile_only_" if args.compile_only else "")
            + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    ).resolve()
    if args.dry_run:
        print(f"Results would be written to: {directory}")
        for repetition in range(1, args.runs + 1):
            for task in args.tasks:
                print(
                    shlex.join(
                        command_for(
                            task,
                            args.seed_base + repetition - 1,
                            directory / "runs" / f"{repetition:03d}_{task}",
                            args,
                        )
                    )
                )
        return 0
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"Choose a new output directory: {directory} already exists")
    (directory / "logs").mkdir()
    (directory / "runs").mkdir()
    manifest = {
        **vars(args),
        "output_dir": str(directory),
        "python": sys.executable,
        "started_at": utc_now(),
        "status": "running",
        "working_directory": str(ROOT),
        "std_ddof": 0,
        "task_modules": {task: MODULES[task] for task in args.tasks},
        "timing": "wall_seconds: child process elapsed; script_seconds: child wall_seconds; "
        "calendar_seconds: clock elapsed including suspend",
    }
    write_json(directory / "manifest.json", manifest)
    print(f"Results: {directory}", flush=True)
    print(
        "Mean/std use successful runs only (ddof=0). CSV/JSON reports refresh after each task.",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    interrupted = False
    save_reports(directory, rows, args.tasks)
    try:
        for repetition in range(1, args.runs + 1):
            for task in args.tasks:
                rows.append(
                    run_one(
                        task,
                        repetition,
                        args.seed_base + repetition - 1,
                        directory,
                        args,
                    )
                )
                save_reports(directory, rows, args.tasks)
                if rows[-1]["status"] == "interrupted":
                    interrupted = True
                    break
            if interrupted:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        save_reports(directory, rows, args.tasks)
        complete = len(rows) == args.runs * len(args.tasks) and all(
            row["status"] == "ok" for row in rows
        )
        manifest.update(
            finished_at=utc_now(),
            status=(
                "interrupted" if interrupted else ("complete" if complete else "failed")
            ),
        )
        write_json(directory / "manifest.json", manifest)
    print(
        f"Per-run results: {directory / 'runs.csv'}\nMeans/stds: {directory / 'result.csv'}",
        flush=True,
    )
    return 130 if interrupted else int(not complete)


if __name__ == "__main__":
    raise SystemExit(main())
