import csv
from pathlib import Path
import tempfile
import unittest

from scripts.latex_export.export_cpu_gpu_tables import (
    SOURCES, duration, latest_results, main, read_run, timing_cells, table, accuracy_cells,
)


class ExportTablesTests(unittest.TestCase):
    def write_run(self, directory, device, seconds=120, tasks=('mnistAdd',)):
        directory.mkdir(parents=True)
        with (directory / 'runs.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['task', 'run', 'status', 'device', 'wall_seconds', 'latent_accuracy_pct',
                             'seed', 'epochs', 'batch_size', 'models_source', 'checkpoint_freq'])
            for task in tasks:
                writer.writerow([task, 1, 'ok', device, seconds, 98.5, 123, 3, 16, 'cache', 0])
        return directory

    def test_speedup_directions_and_duration_rounding(self):
        cells = timing_cells([120, 30, 40, 5])
        self.assertEqual(cells, ['120.00', '30.00', r'4.00$\times$', '40.00',
                                 r'\textbf{5.00}', r'8.00$\times$', r'3.00$\times$', r'6.00$\times$'])
        self.assertEqual(timing_cells([120, 30, 40, 5], True)[0], '2m')
        self.assertEqual(duration(3599.5), '1h')
        self.assertEqual(duration(90061), '25h1m1s')

    def test_reject_incomplete_wrong_device_and_zero_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_run(Path(tmp) / 'run', 'cpu', seconds=0)
            with self.assertRaisesRegex(ValueError, 'invalid wall_seconds'):
                read_run(path, 1, ['mnistAdd'], 'cpu')
            with self.assertRaisesRegex(ValueError, 'expected gpu'):
                read_run(path, 1, ['mnistAdd'], 'gpu')
            self.assertEqual(read_run(path, 1, ['member5'], 'cpu'), {})

    def test_missing_configurations_and_omitted_tasks(self):
        self.assertEqual(timing_cells([120, None, 40, None]),
                         ['120.00', '---', '---', r'\textbf{40.00}', '---', '---',
                          r'3.00$\times$', '---'])
        self.assertEqual(accuracy_cells([98, None, 90, None]),
                         [r'\textbf{98.00}', '---', '90.00', '---'])
        self.assertEqual(timing_cells([None] * 4), ['---'] * 8)
        data = {source: {} for source in SOURCES}
        data['baseline_cpu']['add2x2'] = dict(wall_seconds=60, latent_accuracy_pct=95)
        data['uno_cpu']['member3'] = dict(wall_seconds=None, latent_accuracy_pct=90)
        for accuracy in (False, True):
            content = table(data, ['add2x2', 'member3', 'member5'], 1, accuracy=accuracy)
            self.assertIn('add2x2 &', content)
            self.assertIn('member3 &', content)
            self.assertNotIn('member5 &', content)
            self.assertIn('---', content)
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_run(Path(tmp) / 'run', 'cpu', seconds='')
            self.assertIsNone(read_run(path, 1, ['mnistAdd'], 'cpu')['mnistAdd']['wall_seconds'])

    def test_discovery_and_three_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source, seconds in zip(SOURCES, (120, 30, 40, 5)):
                prefix = source.replace('uno_', 'pipeline_')
                self.write_run(root / f'{prefix}_20261007_010000_000001',
                               'cuda' if source.endswith('gpu') else 'cpu', seconds,
                               tasks=('mnistAdd', 'add2x2') if source == 'baseline_cpu' else ('mnistAdd',))
            self.write_run(root / 'pipeline_cpu_compile_only_20261007_020000_000001', 'cpu')
            self.assertEqual(latest_results(root, 'uno_cpu').name,
                             'pipeline_cpu_20261007_010000_000001')
            output = root / 'export'
            main(['--results-dir', str(root), '--tasks', 'mnistAdd', 'add2x2', 'member5', '--output-dir', str(output)])
            document = (output / 'tables.tex').read_text()
            self.assertEqual(document.count(r'\begin{table}'), 3)
            self.assertIn(r'UNO Speedup', document)
            self.assertIn(r'\textbf{98.50} & \textbf{98.50} & \textbf{98.50} & \textbf{98.50}', document)
            self.assertIn(r'add2x2 & \textbf{120.00} & --- & ---', document)
            self.assertNotIn('member5 &', document)
            self.assertIn(r'Total & \textbf{240.00} & --- & ---', document)
            self.assertEqual(len(list(output.glob('*.tex'))), 6)


if __name__ == '__main__':
    unittest.main()
