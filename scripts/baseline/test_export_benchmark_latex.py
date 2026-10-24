import contextlib
import csv
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import export_benchmark_latex as exporter
else:
    import export_benchmark_latex as exporter


def make_result_dir(parent, name):
    directory = parent / name
    directory.mkdir()
    (directory / 'runs.csv').touch()
    return directory


def write_runs(directory, tasks, accuracy, seconds):
    with (directory / 'runs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['task', 'run', 'status', *exporter.METRICS])
        writer.writeheader()
        for repetition in (1, 2):
            for task in tasks:
                writer.writerow(dict(task=task, run=repetition, status='ok',
                                     downstream_accuracy_pct='',
                                     latent_accuracy_pct=accuracy + repetition,
                                     wall_seconds=seconds * repetition))


class LatexExporterTests(unittest.TestCase):
    def test_partial_results_omit_unrun_tasks_and_keep_missing_cells(self):
        baseline = {'add2x2': [None, 95, 60], 'member5': [None, None, None]}
        pipeline = {'mnistAdd': [90, 98, None]}
        content = exporter.table(baseline, pipeline, 1)
        self.assertIn('add2x2 & --- & --- & 95.00 & --- & 1.00 & ---', content)
        self.assertIn('mnistAdd & --- & 90.00 & --- & 98.00 & --- & ---', content)
        self.assertNotIn('member3 &', content)
        self.assertNotIn('member5 &', content)
        self.assertNotIn('card\\_sum', content)
        self.assertIn('Total & --- & --- & --- & --- & --- & ---', content)
        self.assertIn('add2x2', exporter.TASKS)

    def test_partial_and_empty_runs_are_read_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_runs(directory, ['add2x2'], accuracy=90, seconds=60)
            self.assertEqual(set(exporter.read_run(directory, 1)), {'add2x2'})
            self.assertEqual(exporter.read_run(directory, 3), {})
            with self.assertRaisesRegex(ValueError, 'missing tasks'):
                exporter.read_run(directory, 1, allow_partial=False)
            path = directory / 'runs.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['task', 'run', 'status', *exporter.METRICS])
                writer.writeheader()
                writer.writerow(dict(task='add2x2', run=1, status='ok', latent_accuracy_pct=90))
            content = exporter.table(exporter.read_run(directory, 1), {}, 1)
            self.assertIn('add2x2 & --- & --- & 90.00 & --- & --- & ---', content)

    def test_latest_timestamp_handles_task_names_and_legacy_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            older = make_result_dir(directory, 'pipeline_mnistAdd_20260928_165625_865886')
            latest = make_result_dir(directory, 'pipeline_20260928_180258_465886')
            # Touching an old run must not override its encoded start time.
            os.utime(older / 'runs.csv', (2_000_000_000, 2_000_000_000))
            (directory / 'pipeline_20990101_000000_000000').mkdir()
            (directory / 'pipeline_20990102_000000_000000').write_text('not a directory')
            self.assertEqual(exporter.latest_results(directory, 'pipeline'), latest)

            legacy = make_result_dir(directory, 'baseline_0260922_192544_457308')
            os.utime(legacy / 'runs.csv', (2_000_000_000, 2_000_000_000))
            newer = make_result_dir(directory, 'baseline_20260923_000000_000000')
            self.assertEqual(exporter.latest_results(directory, 'baseline'), newer)
            with self.assertRaisesRegex(ValueError, 'no missing_.*directories'):
                exporter.latest_results(directory, 'missing')

    def test_unknown_directory_names_fall_back_to_csv_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            older = make_result_dir(directory, 'pipeline_z_old')
            newer = make_result_dir(directory, 'pipeline_a_new')
            os.utime(older / 'runs.csv', (1_600_000_000, 1_600_000_000))
            os.utime(newer / 'runs.csv', (1_700_000_000, 1_700_000_000))
            # Directory metadata should not influence this fallback either.
            os.utime(older, (2_000_000_000, 2_000_000_000))
            self.assertEqual(exporter.latest_results(directory, 'pipeline'), newer)

    def test_explicit_inputs_run_two_and_tex_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            baseline = make_result_dir(directory, 'chosen_baseline')
            pipeline = make_result_dir(directory, 'chosen_pipeline')
            write_runs(baseline, exporter.TASKS, accuracy=80, seconds=60)
            write_runs(pipeline, ['mnistAdd'], accuracy=90, seconds=15)
            inputs = {path: path.read_bytes() for path in directory.glob('*/runs.csv')}
            # These auto-selection candidates contain invalid CSVs: overrides must win.
            make_result_dir(directory, 'baseline_20990101_000000_000000')
            make_result_dir(directory, 'pipeline_20990101_000000_000000')
            with patch.object(exporter, 'compile_pdf') as compile_pdf, contextlib.redirect_stdout(io.StringIO()) as captured:
                exporter.main(['--results-dir', str(directory), '--output-dir', str(directory),
                               '--baseline', str(baseline / 'runs.csv'),
                               '--pipeline', str(pipeline), '--run', '2', '--tex-only'])
            compile_pdf.assert_not_called()
            output = directory / 'baseline_vs_pipeline_run2.tex'
            content = output.read_text()
            self.assertIn(f'% Baseline: {baseline / "runs.csv"}', content)
            self.assertIn(f'% Pipeline: {pipeline}', content)
            self.assertIn(r'mnistAdd & --- & --- & 82.00 & \textbf{92.00} & 2.00 & \textbf{0.50}', content)
            self.assertIn('run 2. Single-run values', content)
            self.assertIn('Repetition: 2', captured.getvalue())
            self.assertFalse(output.with_suffix('.pdf').exists())
            self.assertEqual(inputs, {path: path.read_bytes() for path in inputs})


if __name__ == '__main__':
    unittest.main()
