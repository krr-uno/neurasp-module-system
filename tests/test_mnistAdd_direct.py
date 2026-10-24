"""Check direct UNO addition through a single globally enumerated module."""

import copy
import itertools
from pathlib import Path
import re
import tempfile
import unittest

import torch

from examples.mnistAdd_direct.reasoning import (
    DirectAddition,
    build_program,
    compile_direct,
)
from neurasp.module import NeurASPModule


def decimal_labels(totals: list[int], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent Python integer totals as little-endian digits and an upper part."""
    return (
        torch.tensor(
            [[total // 10**column % 10 for column in range(width)] for total in totals]
        ),
        torch.tensor([total // 10**width for total in totals]),
    )


class DirectAdditionTests(unittest.TestCase):
    """Every latent digit assignment belongs to one discovered complete-sum output."""

    single: DirectAddition
    double: DirectAddition

    @classmethod
    def setUpClass(cls) -> None:
        """Compile the small global programs shared by the numerical tests."""
        cls.single, _ = compile_direct(2, 1)
        cls.double, _ = compile_direct(2, 2)

    def test_two_operands_enumerate_all_global_assignments(self) -> None:
        """Two single digits give 100 models; two two-digit operands give 10,000."""
        for reasoner, width, models, outputs in (
            (self.single, 1, 100, 19),
            (self.double, 2, 10000, 199),
        ):
            with self.subTest(width=width):
                self.assertIsInstance(reasoner.module, NeurASPModule)
                self.assertEqual(reasoner.num_operands, 2)
                self.assertEqual(reasoner.num_digits, width)
                self.assertEqual(reasoner.stable_models, models)
                self.assertEqual(reasoner.module.model_count, models)
                self.assertEqual(reasoner.output_count, outputs)
                self.assertEqual(len(reasoner.outputs), outputs)
                self.assertEqual(reasoner.module.models.shape, (models, 2 * width))
                labels, carry = decimal_labels(list(range(outputs)), width)
                columns = reasoner.output_columns(labels, carry)
                self.assertEqual(columns.dtype, torch.long)
                self.assertEqual(columns.device.type, "cpu")
                self.assertEqual(columns.unique().numel(), outputs)
                self.assertEqual(set(columns.tolist()), set(range(outputs)))

    def test_every_single_digit_pair_has_its_exact_sum(self) -> None:
        """All 100 deterministic input pairs put probability one on their sum."""
        assignments = list(itertools.product(range(10), repeat=2))
        digits = torch.tensor(assignments).unsqueeze(-1)
        labels, carry = decimal_labels([sum(pair) for pair in assignments], 1)
        probabilities = torch.nn.functional.one_hot(digits, 10).float()
        columns = self.single.output_columns(labels, carry)
        torch.testing.assert_close(self.single(probabilities, columns), torch.ones(100))

    def test_two_digit_numbers_preserve_positions_and_final_carry(self) -> None:
        """Selected complete numbers include zeros, unequal digits and overflow."""
        operands = [(0, 0), (12, 34), (99, 99), (10, 1), (8, 98), (90, 9), (54, 76)]
        digits = torch.tensor(
            [[[number % 10, number // 10] for number in pair] for pair in operands]
        )
        totals = [sum(pair) for pair in operands]
        labels, carry = decimal_labels(totals, 2)
        probabilities = torch.nn.functional.one_hot(digits, 10).float()
        columns = self.double.output_columns(labels, carry)
        torch.testing.assert_close(
            self.double(probabilities, columns), torch.ones(len(operands))
        )
        wrong_labels, wrong_carry = decimal_labels(
            [(total + 1) % 199 for total in totals], 2
        )
        wrong_columns = self.double.output_columns(wrong_labels, wrong_carry)
        torch.testing.assert_close(
            self.double(probabilities, wrong_columns), torch.zeros(len(operands))
        )

    def test_three_operands_use_one_thousand_global_models(self) -> None:
        """A third operand changes the global assignment and output counts."""
        reasoner, _ = compile_direct(3, 1)
        self.assertEqual(reasoner.stable_models, 1000)
        self.assertEqual(reasoner.output_count, 28)
        self.assertEqual(reasoner.module.models.shape, (1000, 3))
        digits = torch.tensor([[[9], [9], [9]], [[2], [4], [7]], [[0], [0], [0]]])
        labels, carry = decimal_labels([27, 13, 0], 1)
        probabilities = torch.nn.functional.one_hot(digits, 10).float()
        torch.testing.assert_close(
            reasoner(probabilities, reasoner.output_columns(labels, carry)),
            torch.ones(3),
        )

    def test_probability_and_gradient_match_exhaustive_integer_sums(self) -> None:
        """An independent 10,000-assignment oracle agrees on likelihood and gradients."""
        generator = torch.Generator().manual_seed(41)
        logits = torch.randn(2, 2, 2, 10, generator=generator, requires_grad=True)
        probabilities = logits.softmax(-1)
        labels, carry = decimal_labels([137, 18], 2)
        actual = self.double(probabilities, self.double.output_columns(labels, carry))

        assignments = list(itertools.product(range(10), repeat=4))
        totals = torch.tensor([a + 10 * b + c + 10 * d for a, b, c, d in assignments])
        selected = probabilities.flatten(1, 2)[
            :, torch.arange(4), torch.tensor(assignments)
        ]
        weights = selected.prod(-1)
        expected = torch.stack(
            [weights[row, totals == total].sum() for row, total in enumerate((137, 18))]
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
        (actual_gradient,) = torch.autograd.grad(
            -actual.log().mean(), logits, retain_graph=True
        )
        (expected_gradient,) = torch.autograd.grad(-expected.log().mean(), logits)
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=3e-5, atol=3e-6
        )

    def test_cache_preserves_models_and_discovered_output_mapping(self) -> None:
        """The saved global model table and complete-sum outputs reload together."""
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            first, source = compile_direct(2, 1, cache_dir=cache_dir)
            self.assertEqual(source, "clingo")
            cached, source = compile_direct(2, 1, cache_dir=cache_dir)
            self.assertEqual(source, "cache")
            self.assertEqual(first.outputs, cached.outputs)
            self.assertEqual(first.stable_models, cached.stable_models)
            torch.testing.assert_close(first.module.models, cached.module.models)
            labels, carry = decimal_labels(list(range(19)), 1)
            torch.testing.assert_close(
                first.output_columns(labels, carry),
                cached.output_columns(labels, carry),
            )
            rebuilt, source = compile_direct(2, 1, cache_dir=cache_dir, rebuild=True)
            self.assertEqual(source, "clingo")
            self.assertEqual(first.outputs, rebuilt.outputs)
            torch.testing.assert_close(first.module.models, rebuilt.module.models)

    def test_invalid_dimensions_are_rejected_before_enumeration(self) -> None:
        """The direct program requires multiple operands and positive digit width."""
        for operands, width in ((1, 2), (0, 2), (-2, 1), (2, 0), (2, -1)):
            with self.subTest(operands=operands, width=width):
                with self.assertRaises(ValueError):
                    compile_direct(operands, width)

    def test_invalid_shapes_and_unknown_sum_outputs_are_rejected(self) -> None:
        """Labels and neural inputs must match the compiled complete problem."""
        for labels, carry in (
            (torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, dtype=torch.long)),
            (torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)),
            (torch.tensor([[0, 10]]), torch.tensor([0])),
            (torch.tensor([[0, -1]]), torch.tensor([0])),
            (torch.tensor([[0, 0]]), torch.tensor([-1])),
            (torch.tensor([[0, 0]]), torch.tensor([2])),
            (torch.tensor([[9, 9]]), torch.tensor([1])),
        ):
            with self.subTest(labels=labels.tolist(), carry=carry.tolist()):
                with self.assertRaises(ValueError):
                    self.double.output_columns(labels, carry)
        for shape in ((1, 4, 10), (1, 3, 2, 10), (1, 2, 1, 10), (1, 2, 2, 9)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    self.double(torch.full(shape, 0.1), torch.tensor([0]))
        with self.assertRaises(ValueError):
            self.double(torch.full((1, 2, 2, 10), 0.1), torch.tensor([[0]]))

    def test_long_width_program_uses_no_overflowing_decimal_constants(self) -> None:
        """Generating a wide program does not construct whole-number integer powers."""
        program = build_program(2, 40)
        self.assertIsInstance(program, str)
        self.assertIn("nn(", program)
        self.assertNotIn("**", program)
        constants = [int(value) for value in re.findall(r"\b\d+\b", program)]
        self.assertTrue(constants)
        self.assertLess(max(constants), 2**31)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_module_likelihood_and_gradient_match_cpu(self) -> None:
        """A global module moved to CUDA preserves its observed probability path."""
        cuda_reasoner = copy.deepcopy(self.double).to("cuda")
        self.assertTrue(all(buffer.is_cuda for buffer in cuda_reasoner.buffers()))
        generator = torch.Generator().manual_seed(43)
        cpu_logits = torch.randn(2, 2, 2, 10, generator=generator, requires_grad=True)
        cuda_logits = cpu_logits.detach().to("cuda").requires_grad_()
        labels, carry = decimal_labels([47, 146], 2)
        columns = self.double.output_columns(labels, carry)
        cpu_probabilities = self.double(cpu_logits.softmax(-1), columns)
        cuda_probabilities = cuda_reasoner(cuda_logits.softmax(-1), columns)
        torch.testing.assert_close(
            cuda_probabilities.cpu(), cpu_probabilities, rtol=3e-5, atol=1e-7
        )
        (cpu_gradient,) = torch.autograd.grad(
            -cpu_probabilities.log().mean(), cpu_logits
        )
        (cuda_gradient,) = torch.autograd.grad(
            -cuda_probabilities.log().mean(), cuda_logits
        )
        torch.testing.assert_close(
            cuda_gradient.cpu(), cpu_gradient, rtol=5e-4, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
