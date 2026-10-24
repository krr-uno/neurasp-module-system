#!/usr/bin/env python3
"""Compare two benchmark directories in a PDF table (also saves editable LaTeX).

Example:
  python scripts/compare_benchmarks.py --run1 benchmark_results/RUN_A \
    --run2 benchmark_results/RUN_B --title1 GPU --title2 CPU --output comparison.pdf

Read each directory's runs.csv. Titles default to directory names without their
trailing _YYYYMMDD_HHMMSS[_ffffff] timestamp. Incomplete/failed runs are labelled;
only successful repetitions contribute measurements. Multiple measurements show
mean +/- population standard deviation (ddof=0). Rerun to refresh an in-progress
snapshot. Requires pdflatex on PATH; never starts or modifies a benchmark.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("mnistAdd", "member3", "member5") + tuple(
    f"card_{op}_{n}" for op in ("sum", "prodsum") for n in (2, 3, 4)
)
TIME_FIELDS = {
    "wall_seconds": "Total task elapsed time (runner wall_seconds)",
    "script_seconds": "Script elapsed time (script_seconds)",
    "training_seconds": "Training time (training_seconds)",
    "compilation_seconds": "Compilation time (compilation_seconds)",
    "evaluation_seconds": "Evaluation time (evaluation_seconds)",
}
UNITS = {"seconds": (1, "s"), "minutes": (60, "min"), "hours": (3600, "h")}


@dataclass
class Benchmark:
    """A read-only snapshot of one runs.csv and optional invocation manifest."""

    directory: Path
    title: str
    rows: list[dict[str, str]]
    tasks: list[str]
    status: str
    repetitions: int | None


def default_title(directory: Path) -> str:
    """Strip only a trailing benchmark timestamp; preserve the rest of the name."""
    return re.sub(r"_\d{8}_\d{6}(?:_\d{6})?$", "", directory.name) or directory.name


def number(raw: str, field: str) -> float | None:
    """Parse an available nonnegative finite measurement; blanks remain missing."""
    if not raw.strip():
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {field}: {raw!r}")
    if field.endswith("_pct") and value > 100:
        raise ValueError(f"Accuracy exceeds 100: {raw!r}")
    return value


def read_benchmark(directory: Path, title: str | None, time_field: str) -> Benchmark:
    """Read a CSV snapshot and include manifest tasks even when none have finished."""
    directory = directory.resolve()
    path = directory / "runs.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "task",
            "run",
            "status",
            time_field,
            "latent_accuracy_pct",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    seen = set()
    for row in rows:
        if any(row.get(key) is None for key in required) or None in row:
            raise ValueError(f"{path}: malformed CSV row")
        if not row["task"] or not row["run"].isdigit() or int(row["run"]) < 1:
            raise ValueError(f"{path}: invalid task/run identifier")
        identity = (row["task"], int(row["run"]))
        if identity in seen:
            raise ValueError(f"{path}: duplicate task/repetition {identity}")
        seen.add(identity)
        if row["status"] not in ("ok", "failed", "interrupted", "running", "pending"):
            raise ValueError(f"{path}: unknown task status {row['status']!r}")
        if row.get("smoke_test", "").lower() in ("true", "1", "yes"):
            raise ValueError(
                f"{path}: smoke-test results are not benchmark measurements"
            )
        if row["status"] == "ok":
            for field in (time_field, "latent_accuracy_pct"):
                number(row[field], field)

    tasks = list(dict.fromkeys(row["task"] for row in rows))
    status = "unspecified"
    repetitions = None
    manifest = directory / "manifest.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        if not isinstance(metadata, dict):
            raise ValueError(f"{manifest}: expected an object")
        declared = metadata.get("tasks", [])
        if not isinstance(declared, list) or any(
            not isinstance(t, str) for t in declared
        ):
            raise ValueError(f"{manifest}: invalid task list")
        tasks = list(dict.fromkeys([*declared, *tasks]))
        status = str(metadata.get("status", "unspecified"))
        repetitions = metadata.get("runs")
        if repetitions is not None and (
            type(repetitions) is not int or repetitions < 1
        ):
            raise ValueError(f"{manifest}: invalid repetition count")
    return Benchmark(
        directory,
        title if title is not None else default_title(directory),
        rows,
        tasks,
        status,
        repetitions,
    )


def latex(text: str) -> str:
    """Escape titles, task names and filesystem paths as plain LaTeX text."""
    escapes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(c, c) for c in " ".join(text.split()))


def selected_rows(benchmark: Benchmark, task: str) -> list[dict[str, str]]:
    """Return only successful repetitions of a task."""
    return [
        row for row in benchmark.rows if row["task"] == task and row["status"] == "ok"
    ]


def metric_cell(
    rows: list[dict[str, str]], field: str, scale: int = 1, bold: bool = False
) -> str:
    """Format one measurement, or mean/population SD with n for repetitions."""
    values = [v / scale for row in rows if (v := number(row[field], field)) is not None]
    if not values:
        return "---"
    if len(values) == 1:
        value = f"{values[0]:.2f}"
        return rf"\textbf{{{value}}}" if bold else value
    mean = statistics.mean(values)
    deviation = statistics.pstdev(values)
    value = rf"${mean:.2f} \pm {deviation:.2f}$"
    if bold:
        value = rf"$\mathbf{{{mean:.2f} \pm {deviation:.2f}}}$"
    return value + rf" {{\scriptsize(n={len(values)})}}"


def speedup_cell(baseline: float | None, uno: float | None) -> str:
    """Format the ratio of times; missing measurements and zero divisors are undefined."""
    if baseline is None or uno is None or uno == 0:
        return "---"
    return rf"{baseline / uno:.2f}$\times$"


def render(run1: Benchmark, run2: Benchmark, time_field: str, unit: str) -> str:
    """Build the comparison table and source/timing notes."""
    tasks = sorted(
        set(run1.tasks) | set(run2.tasks),
        key=lambda task: (ORDER.index(task) if task in ORDER else len(ORDER), task),
    )
    if not tasks:
        raise ValueError("Neither directory contains or declares any tasks")
    scale, unit_label = UNITS[unit]
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{booktabs,graphicx}",
        r"\usepackage[T1]{fontenc}",
        r"\begin{document}",
        r"\noindent{\large\bfseries Benchmark comparison}\par\medskip",
        r"\noindent\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Task & \multicolumn{2}{c}{Latent accuracy (\%)} & \multicolumn{2}{c}{Time ("
        + unit_label + r")} & Speedup \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & Baseline & UNO & Baseline & UNO & Baseline / UNO \\",
        r"\midrule",
    ]
    task_means: list[list[dict[str, float | None]]] = []
    for task in tasks:
        cells = [latex(task)]
        row_sets = [selected_rows(benchmark, task) for benchmark in (run1, run2)]
        means_by_run: list[dict[str, float | None]] = [{}, {}]
        for field, metric_scale, lower_is_better in (
            ("latent_accuracy_pct", 1, False),
            (time_field, scale, True),
        ):
            values = [
                [
                    value / metric_scale
                    for row in rows
                    if (value := number(row[field], field)) is not None
                ]
                for rows in row_sets
            ]
            means = [statistics.mean(value) if value else None for value in values]
            for index, mean in enumerate(means):
                means_by_run[index][field] = mean
            bold = [False, False]
            if means[0] is not None and means[1] is not None and means[0] != means[1]:
                better = min(means) if lower_is_better else max(means)
                bold = [mean == better for mean in means]
            cells.extend(
                metric_cell(rows, field, metric_scale, bold=bold[index])
                for index, rows in enumerate(row_sets)
            )
        cells.append(speedup_cell(means_by_run[0][time_field], means_by_run[1][time_field]))
        task_means.append(means_by_run)
        lines.append(" & ".join(cells) + r" \\")

    # Give each task equal weight, even if repetition counts differ. Incomplete
    # columns stay missing so totals never compare different task selections.
    summary = ["Summary"]
    totals: list[float | None] = []
    for field in ("latent_accuracy_pct", time_field):
        for index in range(2):
            values = [means[index][field] for means in task_means]
            available = [value for value in values if value is not None]
            aggregate = None
            if len(available) == len(tasks):
                aggregate = (statistics.mean(available) if field == "latent_accuracy_pct"
                             else sum(available))
            summary.append("---" if aggregate is None else f"{aggregate:.2f}")
            if field == time_field:
                totals.append(aggregate)
    summary.append(speedup_cell(*totals))
    lines += [
        r"\midrule",
        " & ".join(summary) + r" \\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\par\medskip\noindent\footnotesize",
        r"\textbf{Legend:} Baseline is run1; UNO is run2.\par",
        latex(TIME_FIELDS[time_field]) + ". Latent accuracy measures image classification.",
        r"Speedup is baseline mean time divided by UNO mean time ($>1$ means UNO is faster).",
        r"Summary shows the unweighted mean of task mean accuracies and the sum of task mean times;",
        r"summary speedup is the ratio of those total times, not the mean of task speedups.",
        r"Summary columns require measurements for every displayed task.",
        r"--- means a measurement is missing or a speedup denominator is zero; it does not mean zero.",
        r"Only successful repetitions are included. Multiple measurements show mean",
        r"$\pm$ population standard deviation (ddof=0), with their count $n$.",
    ]
    lines += [
        latex(
            "Snapshot: "
            + datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S %Z")
        ) + r"\newline"
    ]
    for index, benchmark in enumerate((run1, run2), 1):
        # \newline after path separators allows long absolute source paths to wrap.
        directory = Path(benchmark.directory).resolve()
        shown = directory.relative_to(ROOT) if directory.is_relative_to(ROOT) else directory
        path = latex(str(shown)).replace("/", r"/\allowbreak{}")
        lines.append(rf"Source {index}: \texttt{{{path}}}/runs.csv\par")
    lines += [r"\end{document}"]
    return "\n".join(lines) + "\n"

def compile_pdf(source: str, output: Path, engine: str) -> None:
    """Compile LaTeX every time and replace the existing PDF on success."""
    with tempfile.TemporaryDirectory(prefix="benchmark_comparison_") as directory:
        folder = Path(directory)
        tex = folder / "comparison.tex"
        tex.write_text(source)

        command = [
            engine,
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex.name,
        ]

        result = subprocess.run(
            command,
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise ValueError(
                "LaTeX compilation failed:\n"
                + "\n".join(result.stdout.splitlines()[-20:])
            )

        # Always replace the existing PDF with the newly compiled one.
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".pdf.tmp")
        shutil.copyfile(folder / "comparison.pdf", temporary)
        temporary.replace(output)

        # Always replace the editable LaTeX source as well.
        output.with_suffix(".tex").write_text(source)

def main(argv: Sequence[str] | None = None) -> int:
    """Read two result snapshots and create an editable LaTeX file and compiled PDF."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run1", type=Path, required=True, help="Baseline results directory")
    parser.add_argument("--run2", type=Path, required=True, help="UNO results directory")
    parser.add_argument("--title1", "--run1-title")
    parser.add_argument("--title2", "--run2-title")
    parser.add_argument(
        "--output",
        type=Path,
        help="PDF path; default benchmark_results/comparisons/<names>.pdf",
    )
    parser.add_argument("--time-field", choices=TIME_FIELDS, default="wall_seconds")
    parser.add_argument("--time-unit", choices=UNITS, default="seconds")
    args = parser.parse_args(argv)
    try:
        engine = shutil.which("pdflatex")
        if engine is None:
            raise ValueError(
                "pdflatex is required to generate the PDF; install a LaTeX distribution"
            )
        run1 = read_benchmark(args.run1, args.title1, args.time_field)
        run2 = read_benchmark(args.run2, args.title2, args.time_field)
        filename = (
            default_title(run1.directory) + "_vs_" + default_title(run2.directory)
        )
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
        output = (
            args.output or ROOT / "benchmark_results/comparisons" / (filename + ".pdf")
        ).resolve()
        if output.suffix.lower() != ".pdf":
            raise ValueError("--output must end with .pdf")
        # Keep reports separate from each run's source metrics and other artifacts.
        if any(output.is_relative_to(run.directory) for run in (run1, run2)):
            raise ValueError("Choose an output outside the two input run directories")
        source = render(run1, run2, args.time_field, args.time_unit)
        compile_pdf(source, output, engine)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
    print(f"PDF: {output}\nLaTeX: {output.with_suffix('.tex')}")
    for benchmark in (run1, run2):
        successful = sum(row["status"] == "ok" for row in benchmark.rows)
        print(
            f"{benchmark.title}: {successful} completed task run(s); status {benchmark.status}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
