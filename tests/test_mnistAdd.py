"""Check the MNIST addition example's generated output ordering and gradient path."""

import unittest

import torch

from examples.mnistAdd.mnistAdd import (
    DATA_DIR,
    MNISTTuples,
    accuracy,
    build_module,
    downstream_accuracy,
    mnist,
    normalized_images,
    predict_digits,
)
from examples.mnistAdd.network import Net
from neurasp.module import NeurASPProgram


class AdditionPipelineTests(unittest.TestCase):
    """The MNIST addition example's compiled module and training path."""

    def test_all_digit_pairs_have_correct_sum_columns(self) -> None:
        """Every one-hot digit pair puts all probability on its sum's column."""
        module = build_module()
        self.assertEqual(module.models.shape, (100, 2))
        left = torch.arange(10).repeat_interleave(10)
        right = torch.arange(10).repeat(10)
        inputs = torch.cat(
            [torch.nn.functional.one_hot(x, 10) for x in (left, right)], dim=1
        ).float()
        expected = torch.nn.functional.one_hot(left + right, 19).float()
        torch.testing.assert_close(module(inputs), expected)

    def test_probabilities_and_gradients_match_addition_distribution(self) -> None:
        """Output probabilities and loss gradients equal the directly computed sum
        distribution.
        """
        module = build_module()
        logits = torch.randn(3, 2, 10, requires_grad=True)
        probabilities = logits.softmax(-1)
        actual = module(probabilities.flatten(1))
        expected = torch.stack(
            [
                sum(
                    (
                        probabilities[:, 0, a] * probabilities[:, 1, total - a]
                        for a in range(10)
                        if 0 <= total - a < 10
                    ),
                    torch.zeros(3),
                )
                for total in range(19)
            ],
            dim=1,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(1), torch.ones(3))
        labels = torch.tensor([2, 7, 16])
        loss = -actual[torch.arange(3), labels].log().mean()
        reference = -expected[torch.arange(3), labels].log().mean()
        (gradient,) = torch.autograd.grad(loss, logits, retain_graph=True)
        (expected_gradient,) = torch.autograd.grad(reference, logits)
        torch.testing.assert_close(gradient, expected_gradient)

    def test_generated_module_trains_shared_network(self) -> None:
        """One optimizer step through the program updates the shared digit network."""
        network = Net()
        program = NeurASPProgram([(network, 2)], build_module())
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        before = next(network.parameters()).detach().clone()
        predictions = program([torch.randn(2, 1, 28, 28), torch.randn(2, 1, 28, 28)])
        loss = torch.nn.functional.nll_loss(
            predictions.clamp_min(1e-12).log(), torch.tensor([3, 12])
        )
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, next(network.parameters())))


@unittest.skipUnless((DATA_DIR / "MNIST").exists(), "MNIST is not downloaded")
class MNISTImageTests(unittest.TestCase):
    """The images transformed at once are those torchvision transforms one by one."""

    def test_normalized_images_match_the_transform(self) -> None:
        """normalized_images gives exactly the transform's tensors, and MNISTTuples
        stacks a row's images from them.
        """
        test_split = mnist(train=False)
        images = normalized_images(test_split)
        self.assertEqual(images.shape, (len(test_split), 1, 28, 28))
        self.assertEqual(images.dtype, torch.float32)
        for index in range(0, len(test_split), 50):
            with self.subTest(index=index):
                self.assertTrue(torch.equal(images[index], test_split[index][0]))
        tuples = MNISTTuples(test_split, DATA_DIR / "test_data.txt")
        stacked, label = tuples[0]
        *indices, expected_label = tuples.rows[0].tolist()
        self.assertEqual(label, expected_label)
        self.assertTrue(torch.equal(stacked, images[indices]))

    def test_evaluation_from_one_pass_of_digits(self) -> None:
        """predict_digits gives the network's argmax digit of each image, in batches
        of 1000; with the true digits, the downstream accuracy over the test tuples is
        100%.
        """
        test_split = mnist(train=False)
        images = normalized_images(test_split)[:2500]
        network = Net()
        digits = predict_digits(network, images, torch.device("cpu"))
        with torch.no_grad():
            expected = network(images).argmax(-1)
        self.assertTrue(torch.equal(digits, expected))
        self.assertEqual(
            accuracy(torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 3, 0])), 50.0
        )
        downstream, tuples = downstream_accuracy(
            test_split.targets, test_split, DATA_DIR / "test_data.txt"
        )
        self.assertEqual(downstream, 100.0)
        self.assertEqual(
            tuples, len(MNISTTuples(test_split, DATA_DIR / "test_data.txt"))
        )


if __name__ == "__main__":
    unittest.main()
