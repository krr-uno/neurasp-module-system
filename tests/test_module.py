import itertools
import unittest

import torch
from torch import Tensor, nn, tensor

from neurasp.module import NeurASPModule, NeurASPProgram, semantic_loss


def mnist3_matrix(possible_digits: int, number_of_images: int) -> list[list[int]]:
    """All number_of_images-tuples of digits 0..possible_digits - 1, as lists in
    lexicographic order.
    """
    return [
        list(digits)
        for digits in itertools.product(range(possible_digits), repeat=number_of_images)
    ]


def adjust_indices(matrix: list[list[int]]) -> dict[int, list[Tensor]]:
    """Group digit pairs [a, b] by their sum as NeurASPModule models.

    Each pair becomes the input positions [a, b + adjust], where adjust is the number of
    distinct first digits, so the second image's columns follow the first's.
    """
    adjust = len({row[0] for row in matrix})
    models: dict[int, list[Tensor]] = {}
    for a, b in matrix:
        models.setdefault(a + b, []).append(
            torch.tensor([a, b + adjust], dtype=torch.int32)
        )
    return models


class TestPyTorch(unittest.TestCase):
    """PyTorch behavior the module relies on."""

    def test_prod(self) -> None:
        """Tensor.prod over a dimension, like the product of atom probabilities."""
        a = Tensor([[0.08, 0.8], [0.08, 0.1], [0.65, 0.8]])
        b = a.prod(dim=1)
        torch.testing.assert_close(b, Tensor([0.064, 0.008, 0.52]))
        c = a.unsqueeze(0)
        torch.testing.assert_close(c, Tensor([[[0.08, 0.8], [0.08, 0.1], [0.65, 0.8]]]))
        d = c.prod(dim=2)
        torch.testing.assert_close(d, Tensor([[0.064, 0.008, 0.52]]))


class TestNeurASPModule(unittest.TestCase):
    """NeurASPModule's matrices and forward passes."""

    def test_forward_build_matrices(self) -> None:
        """The buffers for 3x3 sums: the models grouped by output, each output's first
        model and model count; and the projection that forward() builds from them.
        """
        models = adjust_indices(mnist3_matrix(3, 2))
        torch.testing.assert_close(models[0], [tensor([0, 3], dtype=torch.int32)])
        torch.testing.assert_close(
            models[1],
            [tensor([0, 4], dtype=torch.int32), tensor([1, 3], dtype=torch.int32)],
        )
        torch.testing.assert_close(
            models[2],
            [
                tensor([0, 5], dtype=torch.int32),
                tensor([1, 4], dtype=torch.int32),
                tensor([2, 3], dtype=torch.int32),
            ],
        )
        torch.testing.assert_close(
            models[3],
            [tensor([1, 5], dtype=torch.int32), tensor([2, 4], dtype=torch.int32)],
        )
        torch.testing.assert_close(models[4], [tensor([2, 5], dtype=torch.int32)])

        module = NeurASPModule(models, sparse=False)
        index_tensor, projection_tensor = module.models, module.output
        self.assertEqual((module.model_count, module.max_model_count), (9, 3))

        torch.testing.assert_close(
            index_tensor,
            tensor(
                [
                    [0, 3],
                    [0, 4],
                    [1, 3],
                    [0, 5],
                    [1, 4],
                    [2, 3],
                    [1, 5],
                    [2, 4],
                    [2, 5],
                ],
                dtype=torch.int32,
            ),
        )
        # Output o's models are rows model_starts[o] onwards, model_counts[o] of them.
        torch.testing.assert_close(module.model_starts, tensor([0, 1, 3, 6, 8]))
        torch.testing.assert_close(module.model_counts, tensor([1, 2, 3, 2, 1]))
        # projection_tensor needs to be transposed because the original code returns the transpose of the projection tensor as described in the paper. This is because pro() produces a row vector instead of a column vector.
        torch.testing.assert_close(
            projection_tensor.T,
            tensor(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
            ),
        )

    def test_forward_build_matrices_sparse_matches_dense(self) -> None:
        """The sparse and dense projections hold the same values."""
        models = adjust_indices(mnist3_matrix(10, 2))
        dense = NeurASPModule(models, sparse=False)
        sparse = NeurASPModule(models, sparse=True)
        dense_index, dense_projection = dense.models, dense.output
        sparse_index, sparse_projection = sparse.models, sparse.output

        self.assertEqual(torch.strided, dense_projection.layout)
        self.assertEqual(torch.sparse_coo, sparse_projection.layout)
        torch.testing.assert_close(sparse_index, dense_index)
        torch.testing.assert_close(sparse_projection.to_dense(), dense_projection)

    def test_models_must_hold_a_stable_model(self) -> None:
        """No models at all is rejected; an output without models is allowed."""
        cases: list[dict[int, list[torch.Tensor]]] = [{}, {0: [], 1: []}]
        for models in cases:
            with self.subTest(models=models):
                with self.assertRaisesRegex(
                    AssertionError, "at least one stable model"
                ):
                    NeurASPModule(models)
        module = NeurASPModule({0: [], 1: [tensor([0, 1])]})
        torch.testing.assert_close(
            module(torch.tensor([[0.5, 0.5]])), torch.tensor([[0.0, 0.25]])
        )
        torch.testing.assert_close(
            module.forward_single_output(
                torch.tensor([[0.5, 0.5], [0.5, 0.5]]), torch.tensor([0, 1])
            ),
            torch.tensor([0.0, 0.25]),
        )

    def test_model_total_gives_the_padded_result(self) -> None:
        """model_total is the batch's number of models; the flat evaluation it allows
        gives the padded evaluation's probabilities and gradient, also for an output
        without models, and is used without model_total for CPU expected outputs.
        """
        module = NeurASPModule(adjust_indices(mnist3_matrix(10, 2)))
        # Sums 0 and 1 have 1 and 2 models; sum 9 has 10.
        columns = torch.tensor([9, 0, 1, 1])
        self.assertEqual(module.model_total(columns), 15)
        logits = torch.randn(4, 2, 10, requires_grad=True)
        probabilities = logits.softmax(-1).flatten(1)
        flat = module.forward_single_output(
            probabilities, columns, module.model_total(columns)
        )
        padded = module._padded_single_output(
            probabilities, module.model_starts[columns], module.model_counts[columns]
        )
        torch.testing.assert_close(flat, padded)
        # CPU expected outputs give the flat evaluation without model_total.
        torch.testing.assert_close(
            module.forward_single_output(probabilities, columns), flat
        )
        (gradient,) = torch.autograd.grad(flat.sum(), logits, retain_graph=True)
        (expected,) = torch.autograd.grad(padded.sum(), logits)
        torch.testing.assert_close(gradient, expected)
        empty = NeurASPModule({0: [], 1: [tensor([0, 1])]})
        outputs = torch.tensor([1, 0, 1])
        torch.testing.assert_close(
            empty.forward_single_output(
                torch.full((3, 2), 0.5), outputs, empty.model_total(outputs)
            ),
            torch.tensor([0.25, 0.0, 0.25]),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_expected_outputs_with_cuda_input(self) -> None:
        """Expected outputs on the CPU with probabilities on CUDA give the padded
        result of the expected outputs on CUDA.
        """
        module = NeurASPModule(adjust_indices(mnist3_matrix(10, 2))).to("cuda")
        probabilities = torch.randn(4, 2, 10).softmax(-1).flatten(1).to("cuda")
        columns = torch.tensor([9, 0, 18, 1])
        on_cuda = columns.to("cuda")
        expected = module._padded_single_output(
            probabilities, module.model_starts[on_cuda], module.model_counts[on_cuda]
        )
        torch.testing.assert_close(
            module.forward_single_output(probabilities, columns), expected
        )

    def test_semantic_loss_equals_nll_on_the_full_distribution(self) -> None:
        """semantic_loss of the expected outputs' probabilities is NLLLoss on the log of
        forward(), with the same gradient.
        """
        module = NeurASPModule(adjust_indices(mnist3_matrix(10, 2)))
        logits = torch.randn(4, 2, 10, requires_grad=True)
        probabilities = logits.softmax(-1).flatten(1)
        columns = torch.tensor([0, 7, 9, 18])
        loss = semantic_loss(module.forward_single_output(probabilities, columns))
        reference = nn.functional.nll_loss(module(probabilities).log(), columns)
        torch.testing.assert_close(loss, reference)
        (gradient,) = torch.autograd.grad(loss, logits, retain_graph=True)
        (expected,) = torch.autograd.grad(reference, logits)
        torch.testing.assert_close(gradient, expected)

    def test_forward_agrees_for_both_layouts(self) -> None:
        """forward() gives the same result with the sparse and dense projections."""
        models = adjust_indices(mnist3_matrix(10, 2))
        dense_module = NeurASPModule(models, sparse=False)
        sparse_module = NeurASPModule(models, sparse=True)
        input = torch.rand(4, 21)

        torch.testing.assert_close(sparse_module(input), dense_module(input))


class TestNeurASPProgram(unittest.TestCase):
    """How NeurASPProgram applies its networks."""

    def test_networks_are_applied_count_times_in_pair_order(self) -> None:
        """Each (network, count) pair handles count inputs in order; forward,
        expected_outputs and the input-count check.
        """
        torch.manual_seed(0)
        digit = nn.Sequential(nn.Linear(3, 2), nn.Softmax(dim=1))
        trit = nn.Sequential(nn.Linear(4, 3), nn.Softmax(dim=1))
        # Atoms: digit(x0) 0..1, digit(x1) 2..3, trit(x2) 4..6; output = their sum.
        models: dict[int, list[torch.Tensor]] = {total: [] for total in range(5)}
        for d0 in range(2):
            for d1 in range(2):
                for d2 in range(3):
                    models[d0 + d1 + d2].append(tensor([d0, 2 + d1, 4 + d2]))
        module = NeurASPModule(models)
        program = NeurASPProgram([(digit, 2), (trit, 1)], module)
        self.assertEqual(len(program.networks), 2)
        self.assertEqual(program.network_input_count, [2, 1])
        self.assertEqual(program.network_input_starts, [0, 2])
        self.assertEqual(len(list(program.parameters())), 4)
        inputs = [torch.rand(5, 3), torch.rand(5, 3), torch.rand(5, 4)]
        expected = module(
            torch.cat([digit(inputs[0]), digit(inputs[1]), trit(inputs[2])], dim=1)
        )
        torch.testing.assert_close(program(inputs), expected)
        torch.testing.assert_close(program(inputs).sum(dim=1), torch.ones(5))
        columns = torch.tensor([0, 1, 2, 3, 4])
        torch.testing.assert_close(
            program(inputs, expected_outputs=columns),
            expected[torch.arange(5), columns],
        )
        with self.assertRaises(AssertionError):
            program(inputs[:2])

    def test_each_network_is_called_once_on_its_stacked_inputs(self) -> None:
        """A network shared by two inputs runs once on both stacked, with the same
        result as one call per input.
        """
        torch.manual_seed(0)
        digit = nn.Sequential(nn.Linear(3, 2), nn.Softmax(dim=1))
        calls = []
        digit.register_forward_hook(lambda _, args, __: calls.append(args[0].shape))
        # Atoms: digit(x0) 0..1, digit(x1) 2..3; output = their sum.
        models = {
            0: [tensor([0, 2])],
            1: [tensor([0, 3]), tensor([1, 2])],
            2: [tensor([1, 3])],
        }
        program = NeurASPProgram([(digit, 2)], NeurASPModule(models))
        inputs = [torch.rand(5, 3), torch.rand(5, 3)]
        result = program(inputs)
        # One call on both inputs stacked along the batch dimension.
        self.assertEqual(calls, [torch.Size([10, 3])])
        expected = NeurASPModule(models)(
            torch.cat([digit(inputs[0]), digit(inputs[1])], dim=1)
        )
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
