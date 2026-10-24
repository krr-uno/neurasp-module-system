"""Numerical and missing-data checks for the comparison table."""
import csv
from pathlib import Path
import tempfile
import unittest

from scripts import compare_benchmarks as compare


def benchmark(name, measurements):
    rows = [dict(task=task, run=str(index + 1), status=status,
                 latent_accuracy_pct=str(accuracy), wall_seconds=str(seconds))
            for index, (task, accuracy, seconds, status) in enumerate(measurements)]
    return compare.Benchmark(Path('/tmp') / name, name, rows,
                             list(dict.fromkeys(row['task'] for row in rows)), 'complete', None)


class ComparisonTests(unittest.TestCase):
    def test_summary_weights_tasks_equally_and_uses_ratio_of_total_times(self):
        baseline = benchmark('baseline', [('mnistAdd', 70, 60, 'ok'),
                                         ('mnistAdd', 90, 180, 'ok'),
                                         ('member3', 100, 60, 'ok'),
                                         ('member3', 0, 9000, 'failed')])
        uno = benchmark('uno', [('mnistAdd', 90, 30, 'ok'), ('member3', 100, 60, 'ok')])
        source = compare.render(baseline, uno, 'wall_seconds', 'minutes')
        self.assertIn(r'Summary & 90.00 & 95.00 & 3.00 & 1.50 & 2.00$\times$', source)
        task_line = next(line for line in source.splitlines() if line.startswith('mnistAdd &'))
        self.assertIn(r'4.00$\times$', task_line)
        self.assertNotIn('Pipeline', source)
        self.assertNotIn('Down.', source)
        self.assertIn(' & Baseline & UNO & Baseline & UNO & Baseline / UNO', source)

    def test_missing_task_prevents_mismatched_summary_speedup(self):
        baseline = benchmark('baseline', [('mnistAdd', 80, 60, 'ok'), ('member3', 90, 60, 'ok')])
        uno = benchmark('uno', [('mnistAdd', 90, 30, 'ok')])
        source = compare.render(baseline, uno, 'wall_seconds', 'seconds')
        self.assertIn('Summary & 85.00 & --- & 120.00 & --- & ---', source)

    def test_zero_time_and_missing_speedup(self):
        self.assertEqual(compare.speedup_cell(10, 0), '---')
        self.assertEqual(compare.speedup_cell(None, 10), '---')
        self.assertEqual(compare.speedup_cell(10, None), '---')
        self.assertEqual(compare.speedup_cell(0, 10), r'0.00$\times$')

    def test_downstream_accuracy_column_is_not_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with (path / 'runs.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['task', 'run', 'status',
                                                           'wall_seconds', 'latent_accuracy_pct'])
                writer.writeheader()
                writer.writerow(dict(task='mnistAdd', run=1, status='ok',
                                     wall_seconds=60, latent_accuracy_pct=95))
            result = compare.read_benchmark(path, None, 'wall_seconds')
            self.assertEqual(len(result.rows), 1)
            self.assertIn('Summary', compare.render(result, result, 'wall_seconds', 'seconds'))
