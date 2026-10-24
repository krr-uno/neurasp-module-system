"""Batch MNIST tuples while preserving NeurASP's observation/image ordering."""

import torch


def collate_training_tuples(samples):
    """Concatenate each tuple's images along the network's image dimension.

    Member tuples contain three or five images; addition has one per input key.
    NeurASP indexes network outputs by observation, then by image within it.
    """
    inputs, observations = zip(*samples)
    return ({key: torch.cat([data[key] for data in inputs], dim=0)
             for key in inputs[0]}, list(observations))
