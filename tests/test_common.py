"""Check the shared example plumbing: output directories and the metrics files."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import torch

from examples.common import choose_output_dir, write_metrics


class OutputDirectoryTests(unittest.TestCase):
    """The output directory is chosen early and created only with the metrics."""

    def test_directory_is_created_only_when_writing_metrics(self) -> None:
        """Choosing creates nothing; write_metrics creates it with both files."""
        with tempfile.TemporaryDirectory() as directory:
            chosen = choose_output_dir(
                Path(directory) / "run", "task", torch.device("cpu"), 16
            )
            self.assertFalse(chosen.exists())
            write_metrics(chosen, {"task": "task", "accuracy": 1.0})
            self.assertEqual(
                json.loads((chosen / "metrics.json").read_text()),
                {"task": "task", "accuracy": 1.0},
            )
            self.assertTrue((chosen / "runs.csv").exists())

    def test_existing_directory_is_rejected_before_the_run(self) -> None:
        """A taken directory fails when chosen, not after training."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                choose_output_dir(Path(directory), "task", torch.device("cpu"), 16)


if __name__ == "__main__":
    unittest.main()
