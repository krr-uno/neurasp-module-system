"""Shared held-out MNIST task evaluation; independent of either NeurASP engine."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
MEMBER_TEST_SEED = 0
MEMBER_TEST_SIZE = 5000
ADD2X2_TEST_SEED = 0
ADD2X2_TEST_SIZE = 5000


def downstream_cases(task, targets):
    """Return test-image indices, optional queries, labels, and protocol name.

    Addition uses the repository's existing test tuples. Membership uses fixed,
    balanced queries over MNIST test images; these are not paper-provided tuples.
    Add2x2 uses fixed four-image tuples and all four row/column sums.
    Local RandomState leaves training randomness and model seeds untouched.
    """
    targets = np.asarray(targets, dtype=np.int64)
    if task == 'mnistAdd':
        rows = np.loadtxt(ROOT / 'mnistAdd/data/test_data.txt', dtype=np.int64)
        indices, labels = rows[:, :2], rows[:, 2]
        if indices.min() < 0 or indices.max() >= len(targets):
            raise ValueError('Addition test indices do not match the MNIST test split')
        if not np.array_equal(targets[indices].sum(axis=1), labels):
            raise ValueError('Addition test labels do not match the MNIST test split')
        return indices, None, labels, 'mnist_add_existing_test_pairs_v1'
    if task == 'add2x2':
        rng = np.random.RandomState(ADD2X2_TEST_SEED)
        indices = np.stack([rng.choice(len(targets), 4, replace=False)
                            for _ in range(ADD2X2_TEST_SIZE)])
        return indices, None, add2x2_sums(targets[indices]), 'mnist_add2x2_test_seed0_v1'
    if task not in ('member3', 'member5'):
        raise ValueError(f'Unsupported MNIST task: {task}')
    count = int(task[-1])
    rng = np.random.RandomState(MEMBER_TEST_SEED)
    indices = np.empty((MEMBER_TEST_SIZE, count), dtype=np.int64)
    queries = np.empty(MEMBER_TEST_SIZE, dtype=np.int64)
    labels = np.arange(MEMBER_TEST_SIZE) % 2
    for row, label in enumerate(labels):
        indices[row] = rng.choice(len(targets), count, replace=False)
        digits = np.unique(targets[indices[row]])
        candidates = digits if label else np.setdiff1d(np.arange(10), digits)
        queries[row] = rng.choice(candidates)
    return indices, queries, labels, 'mnist_member_balanced_test_seed0_v1'


def add2x2_sums(digits):
    """Return row 1, row 2, column 1, column 2 sums for four digits."""
    return np.column_stack((digits[:, 0] + digits[:, 1],
                            digits[:, 2] + digits[:, 3],
                            digits[:, 0] + digits[:, 2],
                            digits[:, 1] + digits[:, 3]))


def score_downstream(predicted_digits, indices, queries, labels):
    """Apply task rules to argmax digits, rather than marginalize task outputs."""
    selected = np.asarray(predicted_digits)[indices]
    if queries is None and labels.ndim == 2:
        predicted = add2x2_sums(selected)
        correct = (predicted == labels).all(axis=1)
    else:
        predicted = selected.sum(axis=1) if queries is None else (selected == queries[:, None]).any(axis=1)
        correct = predicted == labels
    return float(100 * np.mean(correct))


@torch.no_grad()
def evaluate_downstream(network, mnist_test, task, device):
    """Evaluate only; the fixed inference batch size does not change training."""
    if getattr(mnist_test, 'train', None) is not False:
        raise ValueError('Downstream evaluation requires the held-out MNIST test split')
    indices, queries, labels, protocol = downstream_cases(task, mnist_test.targets)
    # Hash logical tuples, independent of NumPy's binary representation/platform.
    rows = np.column_stack((indices, labels) if queries is None else (indices, queries, labels))
    digest = hashlib.sha256(('\n'.join(' '.join(map(str, row)) for row in rows) + '\n').encode()).hexdigest()
    previous_mode = network.training
    network.eval()
    predictions = []
    try:
        for images, _ in DataLoader(mnist_test, batch_size=1000, shuffle=False):
            predictions.append(network(images.to(device)).argmax(-1).cpu())
    finally:
        network.train(previous_mode)
    accuracy = score_downstream(torch.cat(predictions).numpy(), indices, queries, labels)
    metrics = {'downstream_accuracy_pct': accuracy,
               'downstream_test_examples': len(labels),
               'downstream_protocol': protocol,
               'downstream_test_sha256': digest}
    print(f'Downstream test accuracy: {accuracy:.2f}%')
    print('Downstream evaluation: ' + json.dumps(metrics, sort_keys=True))
    return metrics
