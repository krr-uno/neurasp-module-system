"""Check membership compilation, query rotation, and the training gradient path."""

import unittest

import torch

from examples.members.member import compile_member, membership_probabilities
from examples.mnistAdd.network import Net
from neurasp.module import NeurASPModule, semantic_loss


class MembershipTests(unittest.TestCase):
    def test_probabilities_and_gradients_for_every_query(self) -> None:
        """Both ASP tasks equal the independent-image membership distribution."""
        for count in (3, 5):
            with self.subTest(count=count):
                module = NeurASPModule(compile_member(count)[0].models)
                self.assertEqual(module.model_count, 10**count)
                torch.manual_seed(1)
                logits = torch.randn(
                    10, count, 10, dtype=torch.float64, requires_grad=True
                )
                probabilities = logits.softmax(-1)
                queries = torch.arange(10)
                actual = module(
                    membership_probabilities(probabilities.float(), queries)
                )
                queried = probabilities.gather(
                    2, queries[:, None, None].expand(-1, count, 1)
                ).squeeze(-1)
                absent = (1 - queried).prod(1)
                expected = torch.stack((absent, 1 - absent), dim=1)
                # The full projection sums up to 100,000 products in float32.
                torch.testing.assert_close(
                    actual, expected.float(), rtol=1e-4, atol=1e-5
                )
                labels = queries % 2
                selected = module.forward_single_output(
                    membership_probabilities(probabilities, queries), labels
                )
                torch.testing.assert_close(selected, expected[queries, labels])
                gradient = torch.autograd.grad(
                    semantic_loss(selected), logits, retain_graph=True
                )[0]
                reference = torch.autograd.grad(
                    semantic_loss(expected[queries, labels]), logits
                )[0]
                torch.testing.assert_close(gradient, reference)

    def test_semantic_loss_updates_digit_network(self) -> None:
        """A membership observation trains the shared image classifier."""
        network = Net()
        module = NeurASPModule(compile_member(3)[0].models)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        before = next(network.parameters()).detach().clone()
        probabilities = network(torch.randn(6, 1, 28, 28)).reshape(2, 3, 10)
        selected = module.forward_single_output(
            membership_probabilities(probabilities, torch.tensor([2, 7])),
            torch.tensor([0, 1]),
        )
        semantic_loss(selected).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, next(network.parameters())))


if __name__ == "__main__":
    unittest.main()
