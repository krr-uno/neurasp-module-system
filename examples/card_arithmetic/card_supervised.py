"""Supervised baseline for card arithmetic: train CardNet on the true card classes.

The same network as card_arithmetic.py (CardNet, one shared classifier for every
card) and the same training tuples, but trained on each card's class (the latent
label NeurASP never sees) instead of through the program. Every card image is one
training example, so a variant with N cards has N times as many examples as tuples.
Training uses the same tuple batches as NeurASP, flattened into their card images, so
each step sees the same images; evaluation is card_arithmetic.evaluate, so downstream
and latent accuracy are directly comparable. Run from the repository root:

    python -m examples.card_arithmetic.card_supervised   # sum_2, cpu, batch 16, 10 epochs
    python -m examples.card_arithmetic.card_supervised --variant sum_3 --device gpu
"""

import argparse
from pathlib import Path
import sys
import time

SCRIPT_STARTED = time.perf_counter()

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

# Also allow `python examples/card_arithmetic/card_supervised.py` from any working directory.
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
    card_values,
    evaluate,
)
from examples.card_arithmetic.card_dataset import (
    CardArithmetic,
    add_card_arguments,
    card_images_to,
)
from examples.card_arithmetic.card_network import CardNet
from neurasp.device import autocast, memory_format

# CardNet returns probabilities, so this is NLLLoss over their log, as in
# card_arithmetic.py, but against each card's class instead of the tuple's result.
criterion = nn.NLLLoss()


def train_step(
    network: CardNet,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    cards: torch.Tensor,
    device: torch.device,
    precision: str = "ieee",
) -> torch.Tensor:
    """One optimizer step on a batch of card tuples, images of shape (batch, cards, 3,
    H, W), and their cards' classes, (batch, cards); every card is one example.
    Returns the detached mean loss per card, left on the device.
    """
    # (batch, cards, ...) -> (batch * cards, ...), with the labels in the same order.
    images = card_images_to(images, device).flatten(0, 1)
    labels = cards.to(device, non_blocking=True).flatten()
    optimizer.zero_grad()
    with autocast(device, precision):
        probabilities = network(images)
        loss = criterion(probabilities.clamp_min(1e-12).log(), labels)
    loss.backward()
    optimizer.step()
    return loss.detach()


def main() -> None:
    """Parse the command line, train CardNet on the card classes, evaluate it as
    card_arithmetic.py does and write the metrics.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_card_arguments(parser)
    add_training_arguments(parser, batch_size=16, epochs=10, learning_rate=0.0001)
    args, device = parse_training_arguments(parser)
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be nonnegative")
    task = f"card_{args.variant}_supervised"
    count = int(args.variant.rsplit("_", 1)[1])
    output_dir = choose_output_dir(args.output_dir, task, device, args.batch_size)
    seed_everything(args.seed, args.deterministic)

    network = (
        CardNet().to(device).to(memory_format=memory_format(device, args.precision))
    )
    optimizer = torch.optim.Adam(
        network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Both splits are loaded now, so unknown result labels fail before training.
    result_values = card_result_values(args.variant)
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
    train_cards = count * len(train_dataset)
    dataloader = data_loader(train_dataset, args, device, shuffle=True)

    def step(batch: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        """One training step on a DataLoader batch; returns its loss and number of
        cards.
        """
        images, _, _, cards = batch
        loss = train_step(network, optimizer, images, cards, device, args.precision)
        return loss, cards.numel()

    print(
        f"Start {task}: {args.epochs} epoch(s), {len(train_dataset)} tuples = "
        f"{train_cards} card images, batch size {args.batch_size} tuples, {device}",
        flush=True,
    )
    with timer(device) as training:
        train_epochs(
            network,
            dataloader,
            step,
            epochs=args.epochs,
            examples=train_cards,
            device=device,
            started=SCRIPT_STARTED,
            log_every=100,
        )

    # Built before timing, so evaluation_seconds is the evaluation alone. As in
    # card_arithmetic.py, train accuracy uses the first 1000 training tuples.
    test_loader = data_loader(datasets["test"], args, device)
    train_loader = data_loader(
        Subset(train_dataset, range(min(1000, len(train_dataset)))), args, device
    )
    values = card_values(args.variant)
    with timer(device) as evaluation:
        downstream_accuracy, latent_accuracy = evaluate(
            network, test_loader, values, device
        )
        _, latent_train_accuracy = evaluate(network, train_loader, values, device)
    wall_seconds = time.perf_counter() - SCRIPT_STARTED
    metrics = {
        "task": task,
        **run_metadata(device, args),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        # Card images: N per tuple. The tuples are the same as card_arithmetic.py's.
        "train_examples": train_cards,
        "train_tuples": len(train_dataset),
        "test_examples": len(datasets["test"]),
        "cards_per_example": count,
        "downstream_accuracy_pct": downstream_accuracy,
        "latent_accuracy_pct": latent_accuracy,
        "latent_train_accuracy_pct": latent_train_accuracy,
        "wall_seconds": wall_seconds,
        # Nothing is compiled: these fields keep the columns of card_arithmetic.py.
        "compilation_seconds": None,
        "training_seconds": training.seconds,
        "evaluation_seconds": evaluation.seconds,
        "stable_models": None,
        "projection": None,
        "result_values": result_values,
        **environment_metadata(),
        "notes": "Supervised baseline: CardNet trained with NLL on each card's class "
        "(latent labels), every card image an example. Final weights; downstream uses "
        "arithmetic on each card's argmax class, as card_arithmetic.py. Latent train "
        "accuracy uses the first 1000 training tuples.",
    }
    write_metrics(output_dir, metrics)
    print_summary(output_dir, metrics)


if __name__ == "__main__":
    main()
