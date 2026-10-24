"""Train MNIST Add2x2 with compiled ASP reasoning.

Run from the repository root: python -m examples.add2x2.train
"""

import argparse
import hashlib
import itertools
from pathlib import Path
import sys
import time

SCRIPT_STARTED = time.perf_counter()

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# Also allow `python examples/add2x2/train.py` from any working directory.
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


PROGRAM = """
nn(digit(4,i), [0,1,2,3,4,5,6,7,8,9]).
add2x2(R1,R2,C1,C2) :- digit(1,i,N1), digit(2,i,N2),
    digit(3,i,N3), digit(4,i,N4),
    R1=N1+N2, R2=N3+N4, C1=N1+N3, C2=N2+N4.
"""

# Only reachable observations are outputs; sorted order fixes the label columns.
OUTPUT_SUMS = sorted(
    {
        (a + b, c + d, a + c, b + d)
        for a, b, c, d in itertools.product(range(10), repeat=4)
    }
)
OUTPUT_COLUMNS = {sums: column for column, sums in enumerate(OUTPUT_SUMS)}
OUTPUTS = [f"add2x2({r1},{r2},{c1},{c2})" for r1, r2, c1, c2 in OUTPUT_SUMS]
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
TEST_SIZE = 5000
TEST_SEED = 0


def compile_add2x2(
    options: EnumerationOptions = EnumerationOptions(),
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> tuple[CompiledProgram, str]:
    """Compile all four-digit models, grouped by their observed row/column sums."""
    compiled, source = cached_compile_program(
        PROGRAM, OUTPUTS, options, cache_dir=cache_dir, name="add2x2", rebuild=rebuild
    )
    if [(atom.term, atom.output_count) for atom in compiled.neural_atoms] != [("i", 4)]:
        raise ValueError("Expected one neural input i with four digit outputs")
    return compiled, source


def sums_for(digits: torch.Tensor) -> torch.Tensor:
    """Row 1, row 2, column 1 and column 2 sums of (tuples, 4) digits."""
    return torch.stack(
        (
            digits[:, 0] + digits[:, 1],
            digits[:, 2] + digits[:, 3],
            digits[:, 0] + digits[:, 2],
            digits[:, 1] + digits[:, 3],
        ),
        dim=1,
    )


class Add2x2Tuples(Dataset[tuple[torch.Tensor, int]]):
    """The baseline's fixed 3000 tuples, with one output-column label per item."""

    def __init__(self, mnist: torchvision.datasets.MNIST, examples: Path) -> None:
        """Normalize images once and select the original subset with NumPy seed 1."""
        self.images = normalized_images(mnist)
        rows = np.loadtxt(examples, dtype=np.int64, ndmin=2)
        if rows.shape[1] != 8:
            raise ValueError(f"{examples}: expected four image indices and four sums")
        self.rows = rows[
            np.random.RandomState(1).choice(len(rows), 3000, replace=False)
        ]
        self.labels = [OUTPUT_COLUMNS[tuple(row[4:].tolist())] for row in self.rows]

    def __len__(self) -> int:
        """Return the number of tuples."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return four stacked images (4, 1, 28, 28) and the observed output column."""
        return self.images[self.rows[index, :4].tolist()], self.labels[index]


def mnist(train: bool) -> torchvision.datasets.MNIST:
    """Use the original MNIST examples' shared MNIST image directory."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    return torchvision.datasets.MNIST(
        root=str(DATA_DIR), train=train, download=True, transform=transform
    )


def downstream_cases(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
    """The baseline's seed-0 held-out tuples, labels and logical-tuple SHA-256."""
    rng = np.random.RandomState(TEST_SEED)
    indices = torch.from_numpy(
        np.stack([rng.choice(len(targets), 4, replace=False) for _ in range(TEST_SIZE)])
    )
    labels = sums_for(targets[indices])
    rows = torch.cat((indices, labels), dim=1).tolist()
    digest = hashlib.sha256(
        ("\n".join(" ".join(map(str, row)) for row in rows) + "\n").encode()
    ).hexdigest()
    return indices, labels, digest


def downstream_accuracy(
    digits: torch.Tensor, indices: torch.Tensor, labels: torch.Tensor
) -> float:
    """A held-out tuple is correct only if all four predicted sums match."""
    correct = (sums_for(digits[indices]) == labels).all(1)
    return 100.0 * int(correct.sum()) / len(labels)


def main() -> None:
    """Parse the command line, compile the program, train, evaluate and write the
    metrics.
    """
    task_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_enumeration_arguments(parser)
    add_models_cache_arguments(parser, CACHE_DIR)
    add_training_arguments(parser, batch_size=16, epochs=3, learning_rate=0.001)
    args, device = parse_training_arguments(parser)
    task = "add2x2"
    options = enumeration_options(parser, args)
    output_dir = (
        None
        if args.compile_only
        else choose_output_dir(args.output_dir, task, device, args.batch_size)
    )
    seed_everything(args.seed, args.deterministic)

    with timer() as compilation:
        compiled, models_source = compile_add2x2(
            options,
            CACHE_DIR if args.models_cache else None,
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
    # This environment's torch stubs omit Module.to(memory_format=...), as in members.
    network = Net().to(memory_format=memory_format(device, args.precision))  # type: ignore[call-overload]
    network = network.to(device)
    module = module.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

    mnist_train = mnist(train=True)
    train_dataset = Add2x2Tuples(mnist_train, task_dir / "add2x2_train.txt")
    # Not shuffled, as in the original NeurASP example.
    dataloader = data_loader(train_dataset, args, device)

    def step(batch: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        """One training step on a DataLoader batch; returns its loss and size."""
        images, labels = batch
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device, args.precision):
            probabilities = network(images.flatten(0, 1)).reshape(-1, 40)
            predictions = module.forward_single_output(probabilities, labels)
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
    test_indices, test_labels, test_digest = downstream_cases(mnist_test.targets)
    with timer(device) as evaluation:
        test_digits = predict_digits(network, test_images, device)
        train_digits = predict_digits(network, train_dataset.images[:1000], device)
        latent_accuracy = accuracy(test_digits, mnist_test.targets)
        latent_train_accuracy = accuracy(train_digits, mnist_train.targets[:1000])
        downstream = downstream_accuracy(test_digits, test_indices, test_labels)
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
        "test_examples": TEST_SIZE,
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
        "downstream_protocol": "mnist_add2x2_test_seed0_v1",
        "downstream_test_examples": TEST_SIZE,
        "downstream_test_sha256": test_digest,
        "notes": "Latent accuracy is digit classification on the MNIST test split "
        "(train: its first 1000 images). Downstream applies Add2x2 to argmax "
        "digits on 5000 fixed held-out four-image tuples, with seed 0; all four "
        "row/column sums must match.",
    }
    write_metrics(output_dir, metrics)
    print_summary(output_dir, metrics)


if __name__ == "__main__":
    main()
