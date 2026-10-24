"""Purely neural baseline for card arithmetic: train a network directly on the results.

No program and no card classes: DirectCardNet reads the N card images of a tuple and
predicts its result as a class among the variant's possible results, trained with
cross-entropy on the result labels. It uses the same training and test tuples as
card_arithmetic.py, so their downstream accuracies are comparable. Run from the
repository root:

    python -m examples.card_arithmetic.card_direct   # sum_2, cpu, batch 16, 10 epochs
    python -m examples.card_arithmetic.card_direct --variant sum_3 --device gpu
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

SCRIPT_STARTED = time.perf_counter()

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

# Also allow `python examples/card_arithmetic/card_direct.py` from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.common import (
    add_training_arguments,
    data_loader,
    environment_metadata,
    choose_output_dir,
    parse_training_arguments,
    print_summary,
    run_metadata,
    seed_everything,
    timer,
    train_epochs,
    write_metrics,
)
from examples.card_arithmetic.card_arithmetic import (
    card_result_values,
)
from examples.card_arithmetic.card_dataset import (
    CardArithmetic,
    add_card_arguments,
    card_images_to,
)
from examples.card_arithmetic.card_network import DirectCardNet
from neurasp.device import autocast, memory_format

# The network returns logits over the results, so this is cross-entropy with the
# result's column, the same targets NeurASP's semantic loss uses.
criterion = nn.CrossEntropyLoss()


def train_step(
    network: DirectCardNet,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    columns: torch.Tensor,
    device: torch.device,
    precision: str = "ieee",
) -> torch.Tensor:
    """One optimizer step on a batch of card tuples, images of shape (batch, cards, 3,
    H, W), and the output columns of their results; returns the detached mean loss,
    left on the device.
    """
    images = card_images_to(images, device)
    columns = columns.to(device, non_blocking=True)
    optimizer.zero_grad()
    with autocast(device, precision):
        loss = criterion(network(images), columns)
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def evaluate(
    network: nn.Module,
    dataloader: Iterable[Sequence[torch.Tensor]],
    device: torch.device = torch.device("cpu"),
) -> float:
    """Downstream accuracy, in percent, of network on dataloader's batches of (images,
    columns, results, cards): the share of tuples whose highest-scoring result is the
    right one.
    """
    network.eval()
    correct = examples = 0
    for images, columns, _, _ in dataloader:
        predictions = network(card_images_to(images, device)).argmax(-1)
        correct += int((predictions == columns.to(device)).sum())
        examples += len(columns)
    if examples == 0:
        raise ValueError("Cannot evaluate an empty card dataset")
    return 100.0 * correct / examples


def main() -> None:
    """Parse the command line, train the direct network, evaluate it and write the
    metrics.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_card_arguments(parser)
    add_training_arguments(parser, batch_size=16, epochs=10, learning_rate=0.0001)
    args, device = parse_training_arguments(parser)
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be nonnegative")
    task = f"card_{args.variant}_direct"
    count = int(args.variant.rsplit("_", 1)[1])
    output_dir = choose_output_dir(args.output_dir, task, device, args.batch_size)
    seed_everything(args.seed, args.deterministic)

    # The same result columns as card_arithmetic.py, without compiling the program.
    result_values = card_result_values(args.variant)
    network = (
        DirectCardNet(count, len(result_values))
        .to(device)
        .to(memory_format=memory_format(device, args.precision))
    )
    optimizer = torch.optim.Adam(
        network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Both splits are loaded now, so unknown result labels fail before training.
    datasets = {
        split: CardArithmetic(
            args.image_dir,
            args.variant,
            split,
            result_values,
            preload=args.preload_images,
        )
        for split in ("train", "test")
    }
    train_dataset = datasets["train"]
    dataloader = data_loader(train_dataset, args, device, shuffle=True)

    def step(batch: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        """One training step on a DataLoader batch; returns its loss and size."""
        images, columns, _, _ = batch
        loss = train_step(network, optimizer, images, columns, device, args.precision)
        return loss, len(columns)

    print(
        f"Start {task}: {args.epochs} epoch(s), {len(train_dataset)} tuples, "
        f"{len(result_values)} possible results, batch size {args.batch_size}, "
        f"{device}",
        flush=True,
    )
    with timer(device) as training:
        train_epochs(
            network,
            dataloader,
            step,
            epochs=args.epochs,
            examples=len(train_dataset),
            device=device,
            started=SCRIPT_STARTED,
            log_every=100,
        )

    # Built before timing, so evaluation_seconds is the evaluation alone. Train
    # accuracy uses the first 1000 training tuples, as card_arithmetic.py does.
    test_loader = data_loader(datasets["test"], args, device)
    train_loader = data_loader(
        Subset(train_dataset, range(min(1000, len(train_dataset)))), args, device
    )
    with timer(device) as evaluation:
        downstream_accuracy = evaluate(network, test_loader, device)
        downstream_train_accuracy = evaluate(network, train_loader, device)
    wall_seconds = time.perf_counter() - SCRIPT_STARTED
    metrics: dict[str, object] = {
        "task": task,
        **run_metadata(device, args),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_examples": len(train_dataset),
        "test_examples": len(datasets["test"]),
        "cards_per_example": count,
        "downstream_accuracy_pct": downstream_accuracy,
        "downstream_train_accuracy_pct": downstream_train_accuracy,
        # No card classes are predicted, so there is no latent accuracy.
        "latent_accuracy_pct": None,
        "latent_train_accuracy_pct": None,
        "wall_seconds": wall_seconds,
        # Nothing is compiled: these fields keep the columns of card_arithmetic.py.
        "compilation_seconds": None,
        "training_seconds": training.seconds,
        "evaluation_seconds": evaluation.seconds,
        "stable_models": None,
        "projection": None,
        "result_values": result_values,
        **environment_metadata(),
        "notes": "Purely neural baseline: DirectCardNet trained with cross-entropy on "
        "the result columns; downstream accuracy is the argmax result. Final weights. "
        "Train accuracy uses the first 1000 training tuples.",
    }
    write_metrics(output_dir, metrics)
    print_summary(output_dir, metrics)


if __name__ == "__main__":
    main()
