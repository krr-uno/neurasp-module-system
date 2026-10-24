import unittest

import torch
from torch.utils.data import DataLoader

from examples.baseline.benchmark_batching import collate_training_tuples


class TrainingBatchTests(unittest.TestCase):
    def test_tuple_order_and_partial_final_batch(self):
        for keys, images_per_key in ((('i1', 'i2'), 1), (('i',), 3), (('i',), 5)):
            with self.subTest(images_per_key=images_per_key):
                samples = [({key: torch.full((images_per_key, 1, 28, 28), i)
                             for key in keys}, str(i)) for i in range(18)]
                batches = list(DataLoader(samples, batch_size=16,
                                          collate_fn=collate_training_tuples))
                self.assertEqual([len(obs) for _, obs in batches], [16, 2])
                for data, observations in batches:
                    for key in keys:
                        self.assertEqual(data[key].shape,
                                         (len(observations) * images_per_key, 1, 28, 28))
                        for index, obs in enumerate(observations):
                            images = data[key][index * images_per_key:(index + 1) * images_per_key]
                            self.assertTrue(torch.all(images == int(obs)))


if __name__ == '__main__':
    unittest.main()
