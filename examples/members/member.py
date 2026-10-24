"""Train MNIST membership with compiled ASP reasoning.

Run from the repository root: python -m examples.members.member --count 3
"""

import argparse
from pathlib import Path
import sys
import time

SCRIPT_STARTED = time.perf_counter()

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# Also allow `python examples/members/member.py` from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.common import (
    add_enumeration_arguments,
    add_models_cache_arguments,
    add_training_arguments,
    data_loader,
    environment_metadata,
    choose_output_dir,
    enumeration_options,
    parse_training_arguments,
    print_summary,
    run_metadata,
    seed_everything,
    timer,
    train_epochs,
    write_metrics,
)
from examples.mnistAdd.network import Net
from examples.mnistAdd.mnistAdd import (
    accuracy,
    normalized_images,
    predict_digits,
    MNIST_MEAN,
    MNIST_STD,
)
from neurasp.cache import cached_compile_program
from neurasp.compiler import CompiledProgram, EnumerationOptions
from neurasp.device import autocast, memory_format
from neurasp.module import NeurASPModule, semantic_loss

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def compile_member(
    count: int,
    options: EnumerationOptions = EnumerationOptions(),
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> tuple[CompiledProgram, str]:
    """Compile membership in count digits, with the query normalized to zero.

    Membership is invariant under a permutation of digit names. Training rotates
    each image's probabilities so the queried digit is zero, avoiding ten copies
    of the same stable models for the ten possible queries.
    """
    program = f"""
    nn(digit({count},i), [0,1,2,3,4,5,6,7,8,9]).
    check(0).
    present(D) :- digit(P,i,D), check(D).
    member(D,0) :- check(D), not present(D).
    member(D,1) :- check(D), present(D).
    """
    compiled, source = cached_compile_program(
        program,
        ["member(0,0)", "member(0,1)"],
        options,
        cache_dir=cache_dir,
        name=f"member{count}",
        rebuild=rebuild,
    )
    if [(atom.term, atom.output_count) for atom in compiled.neural_atoms] != [
        ("i", count)
    ]:
        raise ValueError(f"Expected one neural input i with {count} digit outputs")
    return compiled, source


def membership_probabilities(
    probabilities: torch.Tensor, queries: torch.Tensor
) -> torch.Tensor:
    """Rotate (batch, images, digits) probabilities to make each query digit zero."""
    columns = (torch.arange(10, device=probabilities.device) + queries[:, None]) % 10
    return probabilities.gather(
        2, columns[:, None, :].expand_as(probabilities)
    ).flatten(1)


class MembershipTuples(Dataset[tuple[torch.Tensor, int, int]]):
    """Rows of `examples` hold MNIST indices followed by the query and label; an item is the
    stacked images, of shape (images, 1, 28, 28), the query and the label.
    """

    def __init__(self, mnist: torchvision.datasets.MNIST, examples: Path) -> None:
        """Read the rows of the examples file and transform mnist's images once.

        torchvision's MNIST converts and normalizes an image on every access, which
        took 3.0 s of a 10–11 s GPU epoch; the transformed split is 188 MB for the
        60,000 training images.
        """
        self.images = normalized_images(mnist)
        rows = np.loadtxt(examples, dtype=np.int64, ndmin=2)
        # The original examples use the same fixed subset, independent of model seed.
        self.rows = rows[
            np.random.RandomState(1).choice(len(rows), 3000, replace=False)
        ]

    def __len__(self) -> int:
        """Return the number of tuples."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        """Return tuple index as its stacked images, query and label."""
        *indices, query, label = self.rows[index].tolist()
        return self.images[indices], query, label


def mnist(train: bool) -> torchvision.datasets.MNIST:
    """Use the original membership examples' shared MNIST image directory."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    return torchvision.datasets.MNIST(
        root=str(DATA_DIR), train=train, download=True, transform=transform
    )


def downstream_accuracy(
    digits: torch.Tensor, targets: torch.Tensor, count: int
) -> float:
    """Score 5000 fixed balanced test queries, matching the source branch protocol."""
    rng = np.random.RandomState(0)
    indices = np.empty((5000, count), dtype=np.int64)
    queries = np.empty(5000, dtype=np.int64)
    labels = np.arange(5000) % 2
    for row, label in enumerate(labels):
        indices[row] = rng.choice(len(targets), count, replace=False)
        present = np.unique(targets.numpy()[indices[row]])
        candidates = present if label else np.setdiff1d(np.arange(10), present)
        queries[row] = rng.choice(candidates)
    predicted = (
        digits[torch.from_numpy(indices)] == torch.from_numpy(queries)[:, None]
    ).any(1)
    return accuracy(predicted, torch.from_numpy(labels))


def main(count: int | None = None) -> None:
    """Parse the command line, compile the program, train, evaluate and write the
    metrics.
    """
    task_dir = Path(__file__).resolve().parent
    cache_dir = task_dir / ".cache"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    if count is None:
        parser.add_argument("--count", type=int, choices=(3, 5), required=True)
    add_enumeration_arguments(parser)
    add_models_cache_arguments(parser, cache_dir)
    add_training_arguments(parser, batch_size=16, epochs=3, learning_rate=0.001)
    args, device = parse_training_arguments(parser)
    if count is None:
        count = args.count
    task = f"member{count}"
    options = enumeration_options(parser, args)
    output_dir = (
        None
        if args.compile_only
        else choose_output_dir(args.output_dir, task, device, args.batch_size)
    )
    seed_everything(args.seed, args.deterministic)

    with timer() as compilation:
        compiled, models_source = compile_member(
            count,
            options,
            cache_dir if args.models_cache else None,
            args.rebuild_models,
        )
        module = NeurASPModule(compiled.models)
    print(
        f"Models ({models_source}) and module: {compilation.seconds:.3f}s; "
        f"{module.model_count:,} stable models",
        flush=True,
    )
    if output_dir is None:  # --compile-only
        return
    # This environment's torch stubs omit Module.to(memory_format=...).
    network = Net().to(memory_format=memory_format(device, args.precision))  # type: ignore[call-overload]
    network = network.to(device)
    module = module.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

    mnist_train = mnist(train=True)
    train_dataset = MembershipTuples(mnist_train, task_dir / f"{task}_train.txt")
    # Not shuffled, as in the original NeurASP example.
    dataloader = data_loader(train_dataset, args, device)

    def step(batch: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        """One training step on a DataLoader batch; returns its loss and size."""
        images, queries, labels = batch
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()
        queries = queries.to(device, non_blocking=True)
        with autocast(device, args.precision):
            probabilities = network(images.flatten(0, 1)).reshape(-1, count, 10)
            predictions = module.forward_single_output(
                membership_probabilities(probabilities, queries), labels
            )
            loss = semantic_loss(predictions)
        loss.backward()
        optimizer.step()
        return loss.detach(), len(labels)

    print(
        f"Start {task}: {args.epochs} epoch(s), {len(train_dataset)} tuples, "
        f"batch size {args.batch_size}, {device}",
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
            log_every=1000,
        )

    # Loaded before timing, so evaluation_seconds is the evaluation alone, as for the cards.
    # Each image set goes through the network once: the test images' digits give both
    # the latent and the downstream accuracy.
    mnist_test = mnist(train=False)
    test_images = normalized_images(mnist_test)
    with timer(device) as evaluation:
        test_digits = predict_digits(network, test_images, device)
        train_digits = predict_digits(network, train_dataset.images[:1000], device)
        latent_accuracy = accuracy(test_digits, mnist_test.targets)
        latent_train_accuracy = accuracy(train_digits, mnist_train.targets[:1000])
        downstream = downstream_accuracy(test_digits, mnist_test.targets, count)
    wall_seconds = time.perf_counter() - SCRIPT_STARTED

    metrics = {
        "task": task,
        **run_metadata(device, args),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_examples": len(train_dataset),
        # Held-out tuples, as in card_arithmetic; latent accuracy uses latent_test_images.
        "test_examples": 5000,
        "latent_test_images": len(mnist_test),
        "downstream_accuracy_pct": downstream,
        "latent_accuracy_pct": latent_accuracy,
        "latent_train_accuracy_pct": latent_train_accuracy,
        "wall_seconds": wall_seconds,
        "compilation_seconds": compilation.seconds,
        "enumeration": options._asdict(),
        "models_source": models_source,
        "training_seconds": training.seconds,
        "evaluation_seconds": evaluation.seconds,
        "stable_models": module.model_count,
        "projection": "sparse",
        **environment_metadata(),
        "downstream_protocol": "mnist_member_balanced_test_seed0_v1",
        "notes": "Latent accuracy is digit classification on the MNIST test split "
        "(train: its first 1000 images). Downstream applies membership to argmax "
        "digits on 5000 fixed balanced held-out queries, with seed 0.",
    }
    write_metrics(output_dir, metrics)
    print_summary(output_dir, metrics)


if __name__ == "__main__":
    main()
