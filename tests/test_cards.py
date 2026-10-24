"""Bounded native card checks: only two-card modules are enumerated."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from examples.card_arithmetic.card_dataset import CardArithmetic, card_images_to
from neurasp.module import NeurASPModule
from examples.card_arithmetic.card_arithmetic import (
    VARIANTS,
    build_module,
    card_program,
    card_result_values,
    card_values,
    evaluate,
)
from examples.card_arithmetic import card_arithmetic, card_direct, card_supervised
from examples.card_arithmetic.card_network import CardNet, DirectCardNet


class CardModuleTests(unittest.TestCase):
    """The compiled two-card modules and the card program and values."""

    # Variant -> (module, result values), built once for the whole class.
    modules: dict[str, tuple[NeurASPModule, list[int]]]

    @classmethod
    def setUpClass(cls) -> None:
        """Build the two-card sum and prodsum modules once for all tests."""
        cls.modules = {
            variant: build_module(variant) for variant in ("sum_2", "prodsum_2")
        }

    def test_every_two_card_assignment_has_the_correct_result_column(self) -> None:
        """Every one-hot pair of cards puts all probability on its arithmetic result."""
        left = torch.arange(52).repeat_interleave(52)
        right = torch.arange(52).repeat(52)
        inputs = torch.cat(
            [nn.functional.one_hot(card, 52) for card in (left, right)], dim=1
        ).float()
        for variant, (module, outputs) in self.modules.items():
            with self.subTest(variant=variant):
                self.assertEqual(module.models.shape, (52**2, 2))
                self.assertEqual(module.output.layout, torch.sparse_coo)
                values = card_values(variant)
                expected_values = values[left] + values[right]
                output_values = torch.tensor(outputs)
                # Bound temporary tensors while covering every assignment.
                for start in range(0, len(inputs), 64):
                    probabilities = module(inputs[start : start + 64])
                    torch.testing.assert_close(
                        probabilities.sum(1), torch.ones(len(probabilities))
                    )
                    torch.testing.assert_close(
                        output_values[probabilities.argmax(1)],
                        expected_values[start : start + 64],
                    )

    def test_probabilities_and_autograd_match_direct_arithmetic(self) -> None:
        """Output probabilities and loss gradients equal those computed directly from
        the card arithmetic.
        """
        for variant, (module, outputs) in self.modules.items():
            with self.subTest(variant=variant):
                logits = torch.randn(3, 2, 52, requires_grad=True)
                probabilities = logits.softmax(-1)
                actual = module(probabilities.flatten(1))
                values = card_values(variant)
                columns = {value: column for column, value in enumerate(outputs)}
                assignment_columns = torch.tensor(
                    [
                        columns[value]
                        for value in (values[:, None] + values[None, :])
                        .flatten()
                        .tolist()
                    ]
                )
                assignment_probabilities = (
                    probabilities[:, 0, :, None] * probabilities[:, 1, None, :]
                ).flatten(1)
                expected = logits.new_zeros(3, len(outputs)).scatter_add(
                    1, assignment_columns.expand(3, -1), assignment_probabilities
                )
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual.sum(1), torch.ones(3))
                labels = torch.tensor([2, len(outputs) // 2, len(outputs) - 3])
                loss = nn.functional.nll_loss(actual.log(), labels)
                reference_loss = nn.functional.nll_loss(expected.log(), labels)
                (gradient,) = torch.autograd.grad(loss, logits, retain_graph=True)
                (reference_gradient,) = torch.autograd.grad(reference_loss, logits)
                torch.testing.assert_close(gradient, reference_gradient)

    def test_single_output_matches_forward_in_the_expected_column(self) -> None:
        """forward_single_output gives forward()'s value in each expected column,
        padded or flat (with model_total).
        """
        for variant, (module, outputs) in self.modules.items():
            with self.subTest(variant=variant):
                probabilities = torch.randn(4, 2, 52).softmax(-1).flatten(1)
                columns = torch.tensor([0, 2, len(outputs) // 2, len(outputs) - 1])
                expected = module(probabilities)[torch.arange(4), columns]
                actual = module._padded_single_output(
                    probabilities,
                    module.model_starts[columns],
                    module.model_counts[columns],
                )
                torch.testing.assert_close(actual, expected)
                flat = module.forward_single_output(
                    probabilities, columns, module.model_total(columns)
                )
                torch.testing.assert_close(flat, expected)

    def test_all_variants_shift_the_source_program_to_one_based_slots(self) -> None:
        """card_program shifts every variant's slots to 1..N."""
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                count = int(variant.rsplit("_", 1)[1])
                program = card_program(variant)
                self.assertIn("sum(1,V) :- card_value(1,V).", program)
                self.assertIn(f"result(X) :- sum({count},X).", program)
                self.assertNotIn("sum(0,V) :- card_value(0,V).", program)
                self.assertIn(f"nn(card({count},p)", program)

    def test_suit_values_follow_task_files(self) -> None:
        """card_values uses the suit offsets and factors of the task files."""
        self.assertEqual(
            card_values("sum_2")[[0, 13, 26, 39, 51]].tolist(), [41, 15, 28, 2, 14]
        )
        self.assertEqual(
            card_values("prodsum_2")[[0, 13, 26, 39, 51]].tolist(), [8, 4, 6, 2, 14]
        )


class CardDataAndEvaluationTests(unittest.TestCase):
    """The card dataset and the accuracy evaluation."""

    def test_dataset_preserves_card_order_labels_and_validation_image_source(
        self,
    ) -> None:
        """The dataset keeps the CSV's card order, returns the column and the result,
        reads val images from the training directory and rejects unknown results.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks").mkdir()
            (root / "images_train").mkdir()
            (root / "images_train/playing_card_labels.csv").write_text(
                "img,label\n7,2h\n8,ad\n"
            )
            (root / "tasks/card_sum_2_val.csv").write_text(
                "card_1,card_2,result\n7,8,55\n"
            )
            dataset = CardArithmetic(root, "sum_2", "val", [4, 55], root / "tasks")
            image = torch.full((3, 274, 174), 128, dtype=torch.uint8)
            with patch(
                "examples.card_arithmetic.card_dataset.io.read_image",
                return_value=image,
            ) as read_image:
                images, column, result, labels = dataset[0]
            self.assertEqual(
                [Path(call.args[0]).name for call in read_image.call_args_list],
                ["7.jpg", "8.jpg"],
            )
            self.assertEqual(dataset.image_dir, root / "images_train")
            self.assertEqual(images.shape, (2, 3, 274, 174))
            # uint8 as decoded; card_images_to scales them as the original examples.
            self.assertEqual(images.dtype, torch.uint8)
            torch.testing.assert_close(images, torch.full_like(images, 128))
            torch.testing.assert_close(
                card_images_to(images, torch.device("cpu")),
                torch.full((2, 3, 274, 174), 128 / 255.0),
            )
            self.assertEqual(column, 1)
            self.assertEqual(result, 55)
            with self.assertRaisesRegex(
                ValueError, r"Unknown val result labels: \[55\]"
            ):
                CardArithmetic(root, "sum_2", "val", [4, 54], root / "tasks")
            self.assertEqual(labels.tolist(), [0, 51])

    def test_preloaded_images_equal_images_read_from_disk(self) -> None:
        """Preloading decodes each distinct file once and gives the same items."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks").mkdir()
            (root / "images_train").mkdir()
            (root / "images_train/playing_card_labels.csv").write_text(
                "img,label\n7,2h\n8,ad\n9,3c\n"
            )
            # File 7 is used three times, 8 and 9 once each.
            (root / "tasks/card_sum_2_train.csv").write_text(
                "card_1,card_2,result\n7,8,55\n9,7,4\n7,7,4\n"
            )

            def read_image(path: str) -> torch.Tensor:
                """A distinct fake image per file: every value is its number."""
                return torch.full(
                    (3, 274, 174), int(Path(path).stem), dtype=torch.uint8
                )

            with patch(
                "examples.card_arithmetic.card_dataset.io.read_image",
                side_effect=read_image,
            ) as reads:
                lazy = CardArithmetic(root, "sum_2", "train", [4, 55], root / "tasks")
                lazy_items = [lazy[index] for index in range(3)]
                reads.reset_mock()
                eager = CardArithmetic(
                    root, "sum_2", "train", [4, 55], root / "tasks", preload=True
                )
                self.assertEqual(reads.call_count, 3)
                for index, expected in enumerate(lazy_items):
                    actual = eager[index]
                    torch.testing.assert_close(actual[0], expected[0])
                    self.assertEqual(actual[1:3], expected[1:3])
                    torch.testing.assert_close(actual[3], expected[3])
                self.assertEqual(reads.call_count, 3)

    def test_accuracy_counts_argmax_arithmetic_and_individual_cards(self) -> None:
        """evaluate applies arithmetic to argmax cards (not the marginal's argmax) and
        counts latent accuracy per card.
        """

        class FixedNetwork(nn.Module):
            """A stand-in network with fixed outputs."""

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                """Return fixed card probabilities for two tuples of two cards."""
                probabilities = torch.zeros(4, 52)
                probabilities[0, 0] = probabilities[1, 39] = 1.0
                # For the second tuple, each latent argmax is class 0: sum 82.
                # Marginalizing assignments instead gives its highest mass to 43.
                probabilities[2:, 0] = 0.4
                probabilities[2:, 39] = 0.35
                probabilities[2:, 40] = 0.25
                return probabilities

        value_probabilities = {41: 0.4, 2: 0.35, 3: 0.25}
        marginal: dict[int, float] = {}
        for left, p_left in value_probabilities.items():
            for right, p_right in value_probabilities.items():
                marginal[left + right] = (
                    marginal.get(left + right, 0.0) + p_left * p_right
                )
        self.assertEqual(max(marginal, key=lambda total: marginal[total]), 43)
        batch = (
            torch.zeros(2, 2, 3, 1, 1, dtype=torch.uint8),
            torch.tensor([-1, -1]),  # columns: evaluation must not use them
            torch.tensor([43, 43]),
            torch.tensor([[0, 39], [0, 1]]),
        )
        downstream, latent = evaluate(FixedNetwork(), [batch], card_values("sum_2"))
        self.assertEqual(downstream, 50.0)
        self.assertEqual(latent, 75.0)


class DirectBaselineTests(unittest.TestCase):
    """The purely neural baseline: its network, result columns and accuracy."""

    def test_result_values_match_the_compiled_program(self) -> None:
        """card_result_values gives the compiled modules' result columns."""
        for variant in ("sum_2", "prodsum_2"):
            with self.subTest(variant=variant):
                self.assertEqual(card_result_values(variant), build_module(variant)[1])

    def test_card_net_forward_is_softmax_of_its_features(self) -> None:
        """Splitting out features() leaves CardNet's output unchanged."""
        torch.manual_seed(0)
        network = CardNet().eval()
        images = torch.rand(2, 3, 274, 174)
        expected = network.softmax(network.fc2(network.features(images)))
        torch.testing.assert_close(network(images), expected)

    def test_direct_net_concatenates_card_features_in_order(self) -> None:
        """DirectCardNet scores the results from its cards' features, in card order."""
        torch.manual_seed(0)
        network = DirectCardNet(cards=2, outputs=5).eval()
        tuples = torch.rand(3, 2, 3, 274, 174)
        logits = network(tuples)
        self.assertEqual(logits.shape, (3, 5))
        features = [network.encoder(tuples[:, card]) for card in range(2)]
        torch.testing.assert_close(
            logits, network.classifier(torch.cat(features, dim=1))
        )

    def test_accuracy_counts_the_argmax_result(self) -> None:
        """evaluate counts the tuples whose highest-scoring result column is right."""

        class FixedNetwork(nn.Module):
            """A stand-in network with fixed result scores."""

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                """Return scores favoring columns 1, 0 and 2 for three tuples."""
                return torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0, 0, 1.0]])

        batch = (
            torch.zeros(3, 2, 3, 1, 1, dtype=torch.uint8),
            torch.tensor([1, 2, 2]),  # columns
            torch.tensor([-1, -1, -1]),  # results: not used
            torch.zeros(3, 2, dtype=torch.long),  # cards: not used
        )
        accuracy = card_direct.evaluate(FixedNetwork(), [batch])
        self.assertAlmostEqual(accuracy, 200 / 3)


class SupervisedBaselineTests(unittest.TestCase):
    """The supervised baseline's training step on the card classes."""

    def test_train_step_pairs_every_card_image_with_its_class(self) -> None:
        """Each card of each tuple is one example, with its own label; the step
        updates the network.
        """
        torch.manual_seed(0)
        # eval() turns dropout off, so the loss can be recomputed exactly.
        network = CardNet().eval()
        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        # uint8 images, as CardArithmetic returns them; train_step scales them.
        images = torch.randint(0, 256, (2, 3, 3, 274, 174), dtype=torch.uint8)
        scaled = images.float() / 255.0
        cards = torch.tensor([[0, 13, 51], [7, 7, 30]])
        with torch.no_grad():
            expected = sum(
                -network(scaled[tuple_, card : card + 1])[0, cards[tuple_, card]].log()
                for tuple_ in range(2)
                for card in range(3)
            )
        before = network.fc2.weight.detach().clone()
        loss = card_supervised.train_step(
            network, optimizer, images, cards, torch.device("cpu")
        )
        torch.testing.assert_close(loss, expected / 6)
        self.assertFalse(torch.equal(before, network.fc2.weight))


class CardModelsCacheTests(unittest.TestCase):
    """cached_card_models goes through the compile cache entry card_<variant>."""

    def test_cache_entry_per_variant(self) -> None:
        """The first call compiles and writes card_sum_2.pt, the second reads the same
        models and result values from it; without the cache nothing is written.
        """
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(card_arithmetic, "CACHE_DIR", Path(directory)):
                _, _, source = card_arithmetic.cached_card_models("sum_2", cache=False)
                self.assertEqual(source, "clingo")
                self.assertEqual(list(Path(directory).iterdir()), [])
                models, values, source = card_arithmetic.cached_card_models("sum_2")
                self.assertEqual(source, "clingo")
                cached, cached_values, source = card_arithmetic.cached_card_models(
                    "sum_2"
                )
                self.assertEqual(source, "cache")
                self.assertEqual(cached_values, values)
                for column, rows in models.items():
                    torch.testing.assert_close(cached[column], rows)
                self.assertEqual(
                    [path.name for path in Path(directory).iterdir()],
                    ["card_sum_2.pt"],
                )


if __name__ == "__main__":
    unittest.main()
