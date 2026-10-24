"""Check general addition against integer arithmetic and exhaustive latent digits."""

import copy
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from examples.mnistAdd_general.data import (
    AdditionTuples,
    sample_tuples,
    sum_digits,
    tuple_digest,
    validate_dimensions,
)
from examples.mnistAdd_general.reasoning import GeneralAddition, compile_reasoner


def integer_labels(
    operands: list[list[list[int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Python integers to add batches of little-endian decimal operands."""
    width = len(operands[0][0])
    totals = [
        sum(int("".join(map(str, reversed(digits)))) for digits in row)
        for row in operands
    ]
    return (
        torch.tensor(
            [[total // (10**column) % 10 for column in range(width)] for total in totals]
        ),
        torch.tensor([total // (10**width) for total in totals]),
    )


class GeneralAdditionReasoningTests(unittest.TestCase):
    """Compiled local UNO modules give the complete observed sum's likelihood."""

    @classmethod
    def setUpClass(cls) -> None:
        """Reuse the two-operand reasoner across independent numerical checks."""
        cls.reasoner, _ = compile_reasoner(2)

    reasoner: GeneralAddition

    def test_probability_and_gradient_match_exhaustive_integer_addition(self) -> None:
        """Enumerating all 10,000 latent assignments agrees on values and gradients."""
        generator = torch.Generator().manual_seed(8)
        logits = torch.randn(2, 2, 2, 10, generator=generator, requires_grad=True)
        probabilities = logits.softmax(-1)
        labels = torch.tensor([[7, 3], [8, 1]])
        carry = torch.tensor([1, 0])
        actual = self.reasoner(probabilities, labels, carry)

        # The oracle adds complete Python integers. It does not decompose the
        # observations into column sums or infer intermediate carry states.
        assignments = list(itertools.product(range(10), repeat=4))
        totals = torch.tensor([a + 10 * b + c + 10 * d for a, b, c, d in assignments])
        assignments_tensor = torch.tensor(assignments)
        selected = probabilities.flatten(1, 2)[
            :, torch.arange(4), assignments_tensor
        ]
        weights = selected.prod(-1)
        expected = torch.stack(
            [weights[row, totals == total].sum().log() for row, total in enumerate((137, 18))]
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        (actual_gradient,) = torch.autograd.grad(-actual.mean(), logits, retain_graph=True)
        (expected_gradient,) = torch.autograd.grad(-expected.mean(), logits)
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=3e-5, atol=3e-6
        )

    def test_four_eight_and_twelve_operands_propagate_carries(self) -> None:
        """Deterministic wide additions include final carries greater than nine."""
        cases = (
            [[[9, 9, 9], [7, 8, 9], [6, 7, 8], [5, 6, 7]]],
            [[[9, 9, 9]] * 8],
            [[[9, 9]] * 12],
        )
        for digits in cases:
            with self.subTest(operands=len(digits[0])):
                reasoner, _ = compile_reasoner(len(digits[0]))
                probabilities = torch.nn.functional.one_hot(
                    torch.tensor(digits), 10
                ).float()
                labels, carry = integer_labels(digits)
                torch.testing.assert_close(
                    reasoner(probabilities, labels, carry), torch.zeros(1)
                )
                wrong_labels = labels.clone()
                wrong_labels[:, 0] = (wrong_labels[:, 0] + 1) % 10
                self.assertTrue(
                    torch.isneginf(reasoner(probabilities, wrong_labels, carry)).all()
                )
                self.assertEqual(reasoner.num_operands, len(digits[0]))
                self.assertGreater(reasoner.module_count, 0)
                self.assertGreater(reasoner.stable_models, 0)
                self.assertGreater(reasoner.max_module_models, 0)

    def test_four_operand_probability_and_gradient_match_exhaustive_sums(self) -> None:
        """All 10,000 four-digit assignments agree with integer addition."""
        reasoner, _ = compile_reasoner(4)
        generator = torch.Generator().manual_seed(23)
        logits = torch.randn(2, 4, 1, 10, generator=generator, requires_grad=True)
        probabilities = logits.softmax(-1)
        actual = reasoner(
            probabilities, torch.tensor([[3], [8]]), torch.tensor([1, 2])
        )

        assignments = list(itertools.product(range(10), repeat=4))
        totals = torch.tensor([sum(digits) for digits in assignments])
        assignments_tensor = torch.tensor(assignments)
        selected = probabilities.flatten(1, 2)[
            :, torch.arange(4), assignments_tensor
        ]
        weights = selected.prod(-1)
        expected = torch.stack(
            [weights[row, totals == total].sum().log() for row, total in enumerate((13, 28))]
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        (actual_gradient,) = torch.autograd.grad(-actual.mean(), logits, retain_graph=True)
        (expected_gradient,) = torch.autograd.grad(-expected.mean(), logits)
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=3e-5, atol=3e-6
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_likelihoods_and_gradients_match_cpu(self) -> None:
        """Moving compiled modules and carry buffers preserves values and gradients."""
        cpu_reasoner, _ = compile_reasoner(4)
        cuda_reasoner = copy.deepcopy(cpu_reasoner).to("cuda")
        self.assertTrue(all(buffer.is_cuda for buffer in cuda_reasoner.buffers()))
        generator = torch.Generator().manual_seed(29)
        cpu_logits = torch.randn(3, 4, 3, 10, generator=generator, requires_grad=True)
        cuda_logits = cpu_logits.detach().to("cuda").requires_grad_()
        digits = torch.randint(10, (3, 4, 3), generator=generator)
        labels, carry = integer_labels(digits.tolist())

        cpu_likelihoods = cpu_reasoner(cpu_logits.softmax(-1), labels, carry)
        cuda_likelihoods = cuda_reasoner(cuda_logits.softmax(-1), labels, carry)
        torch.testing.assert_close(
            cuda_likelihoods.cpu(), cpu_likelihoods, rtol=3e-5, atol=3e-5
        )
        (cpu_gradient,) = torch.autograd.grad(-cpu_likelihoods.mean(), cpu_logits)
        (cuda_gradient,) = torch.autograd.grad(-cuda_likelihoods.mean(), cuda_logits)
        torch.testing.assert_close(
            cuda_gradient.cpu(), cpu_gradient, rtol=5e-4, atol=1e-5
        )

    def test_long_likelihood_stays_finite_and_has_unclamped_gradient(self) -> None:
        """A 128-digit observation stays differentiable after its probability underflows."""
        generator = torch.Generator().manual_seed(11)
        logits = torch.randn(1, 2, 128, 10, generator=generator, requires_grad=True)
        probabilities = logits.softmax(-1)
        actual = self.reasoner(
            probabilities, torch.zeros(1, 128, dtype=torch.long), torch.zeros(1, dtype=torch.long)
        )
        # Only all-zero operands sum to zero, making this exact without an
        # exponential enumeration or another implementation of the carry algorithm.
        expected = logits.log_softmax(-1)[..., 0].sum((1, 2))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess(actual.item(), -100)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
        (actual_gradient,) = torch.autograd.grad(-actual.sum(), logits, retain_graph=True)
        (expected_gradient,) = torch.autograd.grad(-expected.sum(), logits)
        self.assertTrue(torch.isfinite(actual_gradient).all())
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=3e-5, atol=3e-6
        )

    def test_compiled_modules_round_trip_through_cache(self) -> None:
        """A cache hit preserves likelihoods and rebuild requests compile again."""
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            first, source = compile_reasoner(2, cache_dir=cache_dir)
            self.assertNotEqual(source, "cache")
            cached, source = compile_reasoner(2, cache_dir=cache_dir)
            self.assertEqual(source, "cache")
            self.assertTrue(any(cache_dir.iterdir()))
            probabilities = torch.full((1, 2, 2, 10), 0.1)
            labels, carry = torch.tensor([[3, 7]]), torch.tensor([0])
            torch.testing.assert_close(
                first(probabilities, labels, carry), cached(probabilities, labels, carry)
            )
            self.assertEqual(first.stable_models, cached.stable_models)
            self.assertEqual(first.module_count, cached.module_count)
            rebuilt, source = compile_reasoner(2, cache_dir=cache_dir, rebuild=True)
            self.assertNotEqual(source, "cache")
            torch.testing.assert_close(
                first(probabilities, labels, carry), rebuilt(probabilities, labels, carry)
            )


class GeneralAdditionDataTests(unittest.TestCase):
    """Synthetic MNIST metadata exercises tuple generation without downloading data."""

    def test_dimensions_are_positive_and_require_multiple_operands(self) -> None:
        """At least two operands and one digit are required."""
        validate_dimensions(2, 1)
        validate_dimensions(8, 10)
        for operands, width in ((1, 2), (0, 3), (-2, 2), (2, 0), (2, -1)):
            with self.subTest(operands=operands, width=width):
                with self.assertRaises(ValueError):
                    validate_dimensions(operands, width)

    def test_sampling_is_reproducible_without_consuming_global_rng(self) -> None:
        """A local seed fixes every index independently of training's RNG state."""
        targets = torch.arange(100) % 10
        before = torch.random.get_rng_state().clone()
        first = sample_tuples(targets, 4, 10, 20, seed=17)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        torch.rand(23)
        second = sample_tuples(targets, 4, 10, 20, seed=17)
        other = sample_tuples(targets, 4, 10, 20, seed=18)
        self.assertEqual(first.shape, (20, 4, 10))
        self.assertEqual(first.dtype, torch.long)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertTrue((targets[first][..., -1] != 0).all())

    def test_leading_zero_control_also_applies_to_single_digit_numbers(self) -> None:
        """Zero-only data is usable only when leading zeros are explicitly allowed."""
        targets = torch.zeros(5, dtype=torch.long)
        indices = sample_tuples(targets, 2, 3, 4, seed=7, allow_leading_zeros=True)
        self.assertEqual(indices.shape, (4, 2, 3))
        self.assertTrue((targets[indices] == 0).all())
        for width in (1, 3):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    sample_tuples(targets, 2, width, 4, seed=7)

    def test_labels_use_little_endian_digits_and_arbitrary_integer_width(self) -> None:
        """Exact decimal labels work beyond the range of an int64 whole number."""
        for operands in (
            [[[2, 1], [4, 3]], [[9, 9], [9, 9]]],
            [[[9] * 30] * 3],
        ):
            with self.subTest(width=len(operands[0][0])):
                labels, carry = sum_digits(torch.tensor(operands))
                expected_labels, expected_carry = integer_labels(operands)
                torch.testing.assert_close(labels, expected_labels)
                torch.testing.assert_close(carry, expected_carry)

    def test_dataset_preserves_images_and_their_complete_sum(self) -> None:
        """A tuple returns selected normalized images and their matching sum label."""
        pixels = torch.arange(20, dtype=torch.uint8).view(-1, 1, 1).expand(-1, 28, 28)
        mnist = SimpleNamespace(data=pixels, targets=torch.arange(20) % 10)
        indices = torch.tensor([[[2, 1], [4, 3]], [[9, 9], [9, 9]]])
        dataset = AdditionTuples(mnist, indices)
        self.assertEqual(len(dataset), 2)
        for row in range(2):
            with self.subTest(row=row):
                images, labels, carry = dataset[row]
                self.assertEqual(images.shape, (2, 2, 1, 28, 28))
                expected_images = pixels[indices[row]].unsqueeze(2).float().div(255)
                expected_images = expected_images.sub(0.1307).div(0.3081)
                torch.testing.assert_close(images, expected_images)
                expected_labels, expected_carry = integer_labels(
                    [mnist.targets[indices[row]].tolist()]
                )
                torch.testing.assert_close(labels, expected_labels[0])
                torch.testing.assert_close(carry, expected_carry[0])

    def test_digest_identifies_indices_digits_and_final_carry(self) -> None:
        """Changing any part of a generated tuple changes its reproducibility digest."""
        indices = torch.tensor([[[2, 1], [4, 3]]])
        labels, carry = torch.tensor([[6, 4]]), torch.tensor([0])
        digest = tuple_digest(indices, labels, carry)
        self.assertEqual(digest, tuple_digest(indices.clone(), labels.clone(), carry.clone()))
        self.assertNotEqual(digest, tuple_digest(indices + 1, labels, carry))
        self.assertNotEqual(digest, tuple_digest(indices, labels + 1, carry))
        self.assertNotEqual(digest, tuple_digest(indices, labels, carry + 1))


if __name__ == "__main__":
    unittest.main()
