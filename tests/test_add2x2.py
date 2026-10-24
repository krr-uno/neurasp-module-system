"""Check Add2x2 model grouping, gradients, baseline data and evaluation parity."""

import hashlib
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from examples.add2x2.train import (
    Add2x2Tuples,
    DATA_DIR,
    OUTPUT_COLUMNS,
    OUTPUT_SUMS,
    compile_add2x2,
    downstream_accuracy,
    downstream_cases,
    mnist,
)
from examples.baseline.benchmark_evaluation import downstream_cases as baseline_cases
from examples.mnistAdd.network import Net
from neurasp.module import NeurASPModule, semantic_loss

ROOT = Path(__file__).resolve().parents[1]


class Add2x2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Compile once; reference assignments are independent of the ASP compiler."""
        cls.compiled = compile_add2x2()[0]
        cls.digits = torch.tensor(list(itertools.product(range(10), repeat=4)))
        a, b, c, d = cls.digits.unbind(1)
        cls.labels = torch.tensor(
            [
                OUTPUT_COLUMNS[tuple(row)]
                for row in torch.stack((a + b, c + d, a + c, b + d), 1).tolist()
            ]
        )

    def test_every_assignment_is_grouped_by_all_four_sums(self) -> None:
        """Each digit assignment occurs exactly once under its observed output."""
        module = NeurASPModule(self.compiled.models)
        self.assertEqual(module.model_count, 10000)
        expected = self.digits + torch.arange(4) * 10
        for label, models in self.compiled.models.items():
            self.assertEqual(
                {tuple(row) for row in models.tolist()},
                {tuple(row) for row in expected[self.labels == label].tolist()},
            )
        self.assertEqual(len(self.compiled.models), len(OUTPUT_SUMS))

    def test_observed_probabilities_and_gradients(self) -> None:
        """Semantic loss matches explicit summation of independent digit products."""
        module = NeurASPModule(self.compiled.models)
        torch.manual_seed(1)
        logits = torch.randn(3, 4, 10, dtype=torch.float64, requires_grad=True)
        probabilities = logits.softmax(-1)
        labels = torch.tensor(
            [OUTPUT_COLUMNS[s] for s in ((0, 0, 0, 0), (9, 9, 9, 9), (3, 7, 4, 6))]
        )
        expected = torch.stack(
            [
                probabilities[
                    row, torch.arange(4)[None, :], self.digits[self.labels == label]
                ]
                .prod(1)
                .sum()
                for row, label in enumerate(labels)
            ]
        )
        actual = module.forward_single_output(probabilities.flatten(1), labels)
        torch.testing.assert_close(actual, expected)
        gradient = torch.autograd.grad(
            semantic_loss(actual), logits, retain_graph=True
        )[0]
        reference = torch.autograd.grad(semantic_loss(expected), logits)[0]
        torch.testing.assert_close(gradient, reference)

    def test_shared_network_is_trained(self) -> None:
        """An Add2x2 observation backpropagates to the digit classifier."""
        network = Net()
        module = NeurASPModule(self.compiled.models)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        before = next(network.parameters()).detach().clone()
        probabilities = network(torch.randn(8, 1, 28, 28)).reshape(2, 40)
        labels = torch.tensor(
            [OUTPUT_COLUMNS[(3, 7, 4, 6)], OUTPUT_COLUMNS[(9, 9, 9, 9)]]
        )
        semantic_loss(module.forward_single_output(probabilities, labels)).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, next(network.parameters())))

    def test_compile_cache_reuses_the_same_models(self) -> None:
        """A second compile loads the task's local cache without changing outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            first, source = compile_add2x2(cache_dir=Path(tmp))
            self.assertEqual(source, "clingo")
            second, source = compile_add2x2(cache_dir=Path(tmp))
            self.assertEqual(source, "cache")
            self.assertEqual(first.neural_atoms, second.neural_atoms)
            for label in first.models:
                torch.testing.assert_close(first.models[label], second.models[label])

    def test_downstream_protocol_matches_baseline_and_checks_columns(self) -> None:
        """Both examples score the exact same generated tuples and logical labels."""
        targets = torch.arange(10000) % 10
        indices, labels, digest = downstream_cases(targets)
        baseline_indices, _, baseline_labels, _ = baseline_cases("add2x2", targets)
        np.testing.assert_array_equal(indices.numpy(), baseline_indices)
        np.testing.assert_array_equal(labels.numpy(), baseline_labels)
        rows = np.column_stack((baseline_indices, baseline_labels))
        expected_digest = hashlib.sha256(
            ("\n".join(" ".join(map(str, row)) for row in rows) + "\n").encode()
        ).hexdigest()
        self.assertEqual(digest, expected_digest)
        self.assertEqual(downstream_accuracy(targets, indices, labels), 100)
        digits = torch.tensor([2, 1, 3, 4])
        self.assertEqual(
            downstream_accuracy(
                digits, torch.tensor([[0, 1, 2, 3]]), torch.tensor([[3, 7, 4, 6]])
            ),
            0,
        )

    @unittest.skipUnless((DATA_DIR / "MNIST").exists(), "MNIST is not downloaded")
    def test_training_subset_and_labels_match_baseline(self) -> None:
        """Pipeline tuples preserve the source file, subset, image order and labels."""
        path = ROOT / "examples/add2x2/add2x2_train.txt"
        baseline = ROOT / "examples/baseline/data/add2x2_train.txt"
        self.assertEqual(path.read_bytes(), baseline.read_bytes())
        rows = np.loadtxt(baseline, dtype=np.int64)
        selected = rows[np.random.RandomState(1).choice(len(rows), 3000, replace=False)]
        data = mnist(train=True)
        dataset = Add2x2Tuples(data, path)
        np.testing.assert_array_equal(dataset.rows, selected)
        self.assertEqual(len(dataset), 3000)
        images, label = dataset[0]
        self.assertEqual(tuple(images.shape), (4, 1, 28, 28))
        for slot, index in enumerate(selected[0, :4]):
            torch.testing.assert_close(images[slot], data[int(index)][0])
        self.assertEqual(OUTPUT_SUMS[label], tuple(selected[0, 4:]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
