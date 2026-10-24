import csv
import contextlib
import io
import shlex
import sys
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.baseline import run_benchmarks as runner
from scripts.baseline import export_benchmark_latex as exporter


class BaselineReportingTests(unittest.TestCase):
    def test_all_tasks_parse_downstream_test_only(self):
        for task, *_ in runner.COMMANDS:
            latent = ('98.00\t97.00' if task == 'mnistAdd' else
                      'Test Acc: 97.00%' if task in ('member3', 'member5', 'add2x2') else
                      'Latent test accuracy for card: 97.00%')
            output = 'Downstream validation accuracy: 100.00%\nDownstream test accuracy: 94.00%\n'+latent+'\nDevice: cpu\n'
            metrics = runner.parse_metrics(task, output)
            self.assertEqual(metrics['downstream_accuracy_pct'], 94)
            self.assertEqual(metrics['latent_accuracy_pct'], 97)
            self.assertEqual(metrics['device'], 'cpu')

    def test_metrics_flow_to_summary_csv_and_latex(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for task, *_ in runner.COMMANDS:
                details = dict(downstream_test_examples=5000,
                               downstream_protocol='mnist_member_balanced_test_seed0_v1',
                               downstream_test_sha256='a'*64)
                output = 'Test Acc: 97.00%\nDownstream test accuracy: 94.00%\nDownstream evaluation: '+json.dumps(details)
                metrics = runner.parse_metrics('member3', output)
                rows.append(dict(task=task, run=1, status='ok', exit_code=0, **metrics, log_file='', notes=''))
                rows[-1]['wall_seconds'] = 60
            runner.save_results(root, rows)
            with (root/'result.csv').open() as handle:
                summary = list(csv.DictReader(handle))
            self.assertTrue(all(float(row['downstream_accuracy_pct_mean']) == 94 for row in summary))
            result = exporter.read_run(root, 1)
            table = exporter.table(result, result, 1, exporter.evaluation_note([root], 1))
            self.assertIn('mnistAdd & 94.00 & 94.00 & 97.00 & 97.00 & 1.00 & 1.00', table)
            self.assertIn('5,000 fixed balanced tuples', table)
            self.assertIn('add2x2 & 94.00 & 94.00 & 97.00 & 97.00 & 1.00 & 1.00', table)

    def test_device_forwarding_preserves_batch_arguments(self):
        for device in ('cpu', 'gpu'):
            for spec in runner.COMMANDS:
                command = runner.command_for(spec, device)
                self.assertEqual(command, [sys.executable, *spec[2], '--device', device])
                self.assertNotIn('--batch_size', command)
                self.assertNotIn('--batch-size', command)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main(['--device', 'gpu', '--tasks', 'member3', '--dry-run']), 0)
        self.assertIn('examples/baseline/member3', output.getvalue())
        self.assertIn('--device gpu', output.getvalue())
        self.assertNotIn('card_arithmetic', output.getvalue())

    def test_requested_gpu_rejects_cpu_result(self):
        class Completed:
            returncode = 0
        def fake_run(*args, **kwargs):
            kwargs['stdout'].write('Test Acc: 97.00%\nDownstream test accuracy: 95.00%\nDevice: cpu\n')
            return Completed()
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, 'run', side_effect=fake_run):
            row = runner.run_command(runner.COMMANDS[1], 1, Path(tmp), 'gpu')
            self.assertEqual(row['status'], 'device_mismatch')

    def test_no_cache_forwarded_to_all_tasks(self):
        for spec in runner.COMMANDS:
            self.assertEqual(runner.command_for(spec, 'gpu', no_cache=True),
                             [sys.executable, *spec[2], '--device', 'gpu', '--no-cache'])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main(['--device', 'gpu', '--no-cache', '--dry-run']), 0)
        self.assertEqual(output.getvalue().count('--no-cache'), len(runner.COMMANDS))

    def test_no_cache_reaches_subprocess_and_result_notes(self):
        class Completed:
            returncode = 0
        def fake_run(command, **kwargs):
            self.assertIn('--no-cache', command)
            kwargs['stdout'].write('Test Acc: 97.00%\nDownstream test accuracy: 95.00%\nDevice: cpu\n')
            return Completed()
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, 'run', side_effect=fake_run):
            row = runner.run_command(runner.COMMANDS[1], 1, Path(tmp), no_cache=True)
        self.assertEqual(row['status'], 'ok')
        self.assertIn('Stable-model caching disabled', row['notes'])

    def test_seed_and_checkpoint_forwarding(self):
        for spec in runner.COMMANDS:
            command = runner.command_for(spec, 'gpu', True, 2147483647, 1537081416)
            self.assertEqual(command[command.index('--seed') + 1], '1537081416')
            self.assertIn('--no-cache', command)
            if spec[0].startswith('card_'):
                self.assertEqual(command[command.index('--checkpoint_freq') + 1], '2147483647')
            else:
                self.assertNotIn('--checkpoint_freq', command)
        command = runner.command_for(next(spec for spec in runner.COMMANDS if spec[0] == 'card_sum_2'), 'gpu', checkpoint_freq=0, seed=0)
        self.assertEqual(command[command.index('--seed') + 1], '0')
        self.assertEqual(command[command.index('--checkpoint_freq') + 1], '0')

    def test_selected_gpu_run_passes_options_to_subprocesses(self):
        tasks = ['mnistAdd', 'member3', 'member5', 'add2x2', 'card_sum_2', 'card_sum_3',
                 'card_prodsum_2', 'card_prodsum_3']
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            latent = ('98.00\t97.00' if 'mnistAdd' in command[1] else
                      'Latent test accuracy for card: 97.00%' if '--variant' in command else
                      'Test Acc: 97.00%')
            kwargs['stdout'].write(f'{latent}\nDownstream test accuracy: 95.00%\nDevice: cuda\n')
            return subprocess_result

        subprocess_result = type('Completed', (), {'returncode': 0})()
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, 'run', side_effect=fake_run):
            with contextlib.redirect_stdout(io.StringIO()):
                status = runner.main(['--device', 'gpu', '--no-cache', '--checkpoint_freq=2147483647',
                                      '--seed=1537081416', '--epochs=3', '--batch-size=16', '--output-dir', str(Path(tmp) / 'batch'),
                                      '--tasks', *tasks])
            with (Path(tmp) / 'batch/runs.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(status, 0)
        self.assertEqual([row['task'] for row in rows], tasks)
        self.assertEqual(len(commands), len(tasks))
        for command, row in zip(commands, rows):
            self.assertEqual(command[command.index('--epochs') + 1], '3')
            self.assertEqual(command[command.index('--batch_size') + 1], '16')
            self.assertEqual(command[command.index('--device') + 1], 'gpu')
            self.assertEqual(command[command.index('--seed') + 1], '1537081416')
            self.assertIn('--no-cache', command)
            self.assertIn('Seed: 1537081416', row['notes'])
            if '--variant' in command:
                self.assertEqual(command[command.index('--checkpoint_freq') + 1], '2147483647')
                self.assertIn('Checkpoint frequency: 2147483647', row['notes'])

    def test_missing_downstream_is_not_successful(self):
        class Completed:
            returncode = 0
        def fake_run(*args, **kwargs):
            kwargs['stdout'].write('Test Acc: 97.00%\n')
            return Completed()
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, 'run', side_effect=fake_run):
            row = runner.run_command(runner.COMMANDS[1], 1, Path(tmp))
            self.assertEqual(row['status'], 'missing_metrics')


if __name__ == '__main__':
    unittest.main()
