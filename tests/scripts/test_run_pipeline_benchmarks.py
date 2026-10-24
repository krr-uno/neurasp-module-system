"""Check the sequential runner with small real child processes, not training jobs."""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import Mock, patch

from scripts import run_pipeline_benchmarks as runner

STUB = """
import csv, json, sys
from pathlib import Path
task, seed, output, batch, device, precision = sys.argv[1:]
seed = int(seed)
output = Path(output)
assert not output.exists(), "the runner must not pre-create task output directories"
# Reports must have been saved before starting the next child.
previous = json.loads((output.parents[1]/'runs.json').read_text())
assert len(previous) == (seed-1)*3 + ['mnistAdd','card_sum_2','card_sum_3'].index(task)
if task == 'card_sum_2' and seed == 1:
    print('intentional failure', flush=True)
    sys.exit(3)
output.mkdir()
metrics = dict(task=task, run=1, seed=seed, status='ok', device=device,
    precision=precision, deterministic=False, batch_size=int(batch), epochs=1,
    learning_rate=.001, wall_seconds=1.0, compilation_seconds=.1,
    training_seconds=.5, evaluation_seconds=.2, models_source='clingo',
    downstream_accuracy_pct=80+seed, latent_accuracy_pct=90+seed)
(output/'metrics.json').write_text(json.dumps(metrics))
with (output/'runs.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(metrics))
    writer.writeheader()
    writer.writerow(metrics)
print('child output captured', flush=True)
"""


class RunnerTests(unittest.TestCase):
    """Task routing, reports, interruption, and invalid-input regressions."""

    def test_add2x2_is_selected_explicitly_and_by_default(self) -> None:
        self.assertIn("add2x2", runner.parse_args([]).tasks)
        args = runner.parse_args(["--tasks", "add2x2", "--compile-only"])
        self.assertEqual(args.tasks, ["add2x2"])
        command = runner.command_for("add2x2", 1, Path("/tmp/add2x2"), args)
        self.assertEqual(command[command.index("-m") + 1], "examples.add2x2.train")
        self.assertIn("--compile-only", command)
        self.assertNotIn("--variant", command)

    def test_compile_only_reports_without_metrics(self) -> None:
        """Compilation successes need no metrics; child failures still fail the run."""
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "compile"
                native_command = runner.command_for

                def command(task, seed, path, args):
                    native = native_command(task, seed, path, args)
                    self.assertIn("--compile-only", native)
                    self.assertIn("--models-cache", native)
                    self.assertIn("--rebuild-models", native)
                    return [sys.executable, "-c", "import sys; sys.exit(" +
                            str(3 if failure and task == "member3" else 0) + ")"]

                with patch.object(runner, "command_for", side_effect=command), \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = runner.main([
                        "--tasks", "mnistAdd", "member3", "card_sum_2",
                        "--compile-only", "--rebuild-models", "--output-dir", str(output),
                    ])
                self.assertEqual(code, int(failure))
                rows = json.loads((output / "runs.json").read_text())
                self.assertEqual([r["status"] for r in rows],
                                 ["ok", "failed" if failure else "ok", "ok"])
                for row in rows:
                    self.assertIsNone(row["metrics_file"])
                    self.assertIsNone(row["training_seconds"])
                    self.assertIsNone(row["downstream_accuracy_pct"])
                    self.assertGreaterEqual(row["wall_seconds"], 0)
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertTrue(manifest["compile_only"])
                self.assertEqual(manifest["status"], "failed" if failure else "complete")
                summary = json.loads((output / "result.json").read_text())[0]
                self.assertEqual(summary["downstream_accuracy_pct_n"], 0)
                self.assertIsNone(summary["training_seconds_mean"])

    def test_baseline_settings_and_no_cache(self) -> None:
        tasks = ["mnistAdd", "member3", "member5", "add2x2", "card_sum_2", "card_sum_3"]
        args = runner.parse_args([
            "--tasks", *tasks, "--device", "gpu", "--epochs", "3",
            "--batch-size", "16", "--seed", "1537081416", "--no-cache",
            "--runs", "1",
        ])
        self.assertTrue(runner.parse_args([]).models_cache)
        self.assertEqual(args.seed_base, 1537081416)
        self.assertFalse(args.models_cache)
        for task in tasks:
            command = runner.command_for(task, args.seed_base, Path("/tmp/run"), args)
            for option, value in (("--epochs", "3"), ("--batch-size", "16"),
                                  ("--seed", "1537081416"), ("--device", "cuda")):
                self.assertEqual(command[command.index(option) + 1], value)
            self.assertIn("--no-models-cache", command)
            self.assertNotIn("--models-cache", command)

    def test_subprocess_order_repetitions_failure_and_statistics(self) -> None:
        """Real children verify reports are incremental; one failure doesn't stop later tasks."""

        def command(
            task: str, seed: int, output: Path, args: argparse.Namespace
        ) -> list[str]:
            """Replace training with a tiny process writing the same metric contract."""
            return [
                sys.executable,
                "-B",
                "-u",
                "-c",
                STUB,
                task,
                str(seed),
                str(output),
                str(args.batch_size),
                args.device,
                args.precision,
            ]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            with patch.object(
                runner, "command_for", side_effect=command
            ), contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(
                    [
                        "--runs",
                        "2",
                        "--device",
                        "gpu",
                        "--tasks",
                        "mnistAdd",
                        "card_sum_2",
                        "card_sum_3",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(code, 1)
            rows = json.loads((output / "runs.json").read_text())
            self.assertEqual(
                [r["task"] for r in rows], ["mnistAdd", "card_sum_2", "card_sum_3"] * 2
            )
            self.assertEqual([r["run"] for r in rows], [1, 1, 1, 2, 2, 2])
            self.assertEqual([r["seed"] for r in rows], [1, 1, 1, 2, 2, 2])
            self.assertEqual(rows[1]["status"], "failed")
            self.assertEqual(rows[1]["exit_code"], 3)
            self.assertIsNone(rows[1]["latent_accuracy_pct"])
            for r in rows:
                self.assertTrue(Path(r["log_file"]).exists())
                if r["status"] == "ok":
                    self.assertIn(
                        "child output captured", Path(r["log_file"]).read_text()
                    )
                    self.assertEqual(r["script_seconds"], 1.0)
                    self.assertEqual(r["device"], "cuda")
                    self.assertTrue(Path(r["metrics_file"]).exists())
            with (output / "runs.csv").open() as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[1]["latent_accuracy_pct"], "")
            summaries = json.loads((output / "result.json").read_text())
            self.assertEqual(summaries[0]["latent_accuracy_pct_mean"], 91.5)
            self.assertEqual(summaries[0]["latent_accuracy_pct_std"], 0.5)
            self.assertEqual(summaries[1]["successful_runs"], 1)
            self.assertEqual(summaries[1]["failed_runs"], 1)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["requested_device"], "gpu")

    def test_dry_run_all_tasks_and_forwarded_flags(self) -> None:
        """Every task maps to an actual entry point; dry runs create no output."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dry"
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                code = runner.main(
                    [
                        "--dry-run",
                        "--runs",
                        "2",
                        "--device",
                        "gpu",
                        "--batch-size",
                        "32",
                        "--precision",
                        "bf16",
                        "--no-models-cache",
                        "--compile-processes",
                        "4",
                        "--workers",
                        "2",
                        "--preload-images",
                        "--image-dir",
                        tmp,
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            commands = [
                shlex.split(line)
                for line in capture.getvalue().splitlines()
                if " -m examples." in line
            ]
            self.assertEqual(len(commands), 2 * len(runner.TASKS))
            for index, cmd in enumerate(commands):
                task = runner.TASKS[index % len(runner.TASKS)]
                self.assertEqual(cmd[cmd.index("-m") + 1], runner.MODULES[task])
                self.assertEqual(cmd[cmd.index("--device") + 1], "cuda")
                self.assertEqual(cmd[cmd.index("--batch-size") + 1], "32")
                self.assertEqual(cmd[cmd.index("--precision") + 1], "bf16")
                self.assertEqual(cmd[cmd.index("--seed") + 1], str(index // len(runner.TASKS) + 1))
                self.assertIn("--no-models-cache", cmd)
                self.assertNotIn("--epochs", cmd)
                self.assertTrue(
                    (
                        runner.ROOT / (runner.MODULES[task].replace(".", "/") + ".py")
                    ).is_file()
                )
                if task.startswith("card_"):
                    self.assertEqual(
                        cmd[cmd.index("--variant") + 1], task.removeprefix("card_")
                    )
                    self.assertIn("--preload-images", cmd)
                    self.assertIn("--image-dir", cmd)
                else:
                    self.assertNotIn("--preload-images", cmd)
                    self.assertNotIn("--image-dir", cmd)

    def test_reject_bad_metrics(self) -> None:
        """Bad identities, devices, values or smoke outputs cannot enter summaries."""
        args = runner.parse_args([])
        good: dict[str, Any] = dict(
            task="mnistAdd",
            seed=1,
            status="ok",
            device="cpu",
            precision="ieee",
            deterministic=False,
            batch_size=16,
            epochs=1,
            wall_seconds=2,
            compilation_seconds=0.1,
            training_seconds=1,
            evaluation_seconds=0.2,
            latent_accuracy_pct=98,
            downstream_accuracy_pct=97,
        )
        changes: list[dict[str, Any]] = [
            {"device": "cuda"},
            {"task": "card_sum_2"},
            {"seed": 2},
            {"precision": "bf16"},
            {"batch_size": 32},
            {"smoke_test": True},
            {"latent_accuracy_pct": float("nan")},
            {"wall_seconds": -1},
            {"downstream_accuracy_pct": None},
            {"latent_accuracy_pct": 101},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "metrics.json"
            for change in changes:
                with self.subTest(change=change):
                    p.write_text(json.dumps({**good, **change}))
                    with self.assertRaises(ValueError):
                        runner.read_metrics(p, "mnistAdd", 1, args)
            p.write_text(json.dumps(good))
            self.assertEqual(runner.read_metrics(p, "mnistAdd", 1, args), good)

    def test_interrupt_stops_child_and_saves_status(self) -> None:
        """Ctrl-C reaps the active process group and prevents starting another task."""
        process = Mock(pid=123, returncode=-2)
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt(), -2]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "interrupted"
            with patch.object(
                runner.subprocess, "Popen", return_value=process
            ) as spawn, patch.object(
                runner.os, "killpg"
            ) as kill, contextlib.redirect_stdout(
                io.StringIO()
            ):
                code = runner.main(
                    ["--tasks", "mnistAdd", "card_sum_2", "--output-dir", str(output)]
                )
            self.assertEqual(code, 130)
            spawn.assert_called_once()
            kill.assert_called_once()
            rows = json.loads((output / "runs.json").read_text())
            self.assertEqual(rows[0]["status"], "interrupted")
            self.assertEqual(
                json.loads((output / "manifest.json").read_text())["status"],
                "interrupted",
            )
            self.assertEqual(process.wait.call_count, 2)

    def test_existing_output_and_unavailable_tasks_rejected(self) -> None:
        """Never reuse results, silently omit requested tasks, or accept invalid options."""
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "keep.txt"
            marker.write_text("keep")
            with self.assertRaises(SystemExit):
                runner.main(["--output-dir", tmp])
            self.assertEqual(marker.read_text(), "keep")
        for argv in (
            ["--tasks", "missing_task"],
            ["--runs", "0"],
            ["--batch-size", "0"],
            ["--device", "cpu", "--precision", "bf16"],
            ["--workers", "-1"],
            ["--runs", "2", "--seed-base", str(2**32 - 1)],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                runner.parse_args(argv)

    def test_explicit_overrides_and_deduplicated_order(self) -> None:
        """Keep user task order and pass overrides without changing child defaults implicitly."""
        args = runner.parse_args(
            [
                "--tasks",
                "card_sum_3",
                "mnistAdd",
                "card_sum_3",
                "--epochs",
                "2",
                "--lr",
                ".0002",
                "--deterministic",
                "--rebuild-models",
            ]
        )
        self.assertEqual(args.tasks, ["card_sum_3", "mnistAdd"])
        cmd = runner.command_for("mnistAdd", 3, Path("/tmp/test-output"), args)
        for option, value in (("--epochs", "2"), ("--learning-rate", "0.0002")):
            self.assertEqual(cmd[cmd.index(option) + 1], value)
        self.assertIn("--deterministic", cmd)
        self.assertIn("--rebuild-models", cmd)


if __name__ == "__main__":
    unittest.main()
