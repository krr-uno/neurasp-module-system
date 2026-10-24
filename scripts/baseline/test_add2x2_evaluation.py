import unittest

import numpy as np

from examples.baseline.benchmark_evaluation import downstream_cases, score_downstream


class Add2x2EvaluationTests(unittest.TestCase):
    def test_fixed_held_out_tuples_and_random_state(self):
        targets = np.arange(10000) % 10
        np.random.seed(123)
        before = np.random.get_state()
        indices, queries, labels, protocol = downstream_cases('add2x2', targets)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(indices.shape, (5000, 4))
        self.assertEqual(labels.shape, (5000, 4))
        self.assertIsNone(queries)
        self.assertEqual(protocol, 'mnist_add2x2_test_seed0_v1')
        self.assertTrue(all(len(set(row)) == 4 for row in indices))
        repeated = downstream_cases('add2x2', targets)
        np.testing.assert_array_equal(indices, repeated[0])
        np.testing.assert_array_equal(labels, repeated[2])
        self.assertEqual(score_downstream(targets, indices, queries, labels), 100)

    def test_all_four_sums_must_match(self):
        indices = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        digits = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        labels = np.array([[3, 7, 4, 6], [3, 7, 4, 6]])
        self.assertEqual(score_downstream(digits, indices, None, labels), 100)
        # Swapping the first two digits preserves row sums but changes columns.
        digits[:2] = [2, 1]
        self.assertEqual(score_downstream(digits, indices, None, labels), 50)

    def test_existing_addition_and_membership_scores(self):
        digits = np.array([1, 2, 3])
        indices = np.array([[0, 1], [1, 2]])
        self.assertEqual(score_downstream(digits, indices, None, np.array([3, 5])), 100)
        self.assertEqual(score_downstream(digits, indices, np.array([1, 0]), np.array([1, 0])), 100)


if __name__ == '__main__':
    unittest.main()
