"""Train MNIST addition with the neurasp parser, generator and NeurASPProgram.

The NeurASP program is compiled once into a NeurASPModule (all stable models
enumerated by Clingo); training then runs in PyTorch alone. Run from the
repository root:

    python -m examples.mnistAdd.mnistAdd                  # cpu, batch 16, 1 epoch
    python -m examples.mnistAdd.mnistAdd --device gpu
"""

import argparse
from pathlib import Path
import sys
import time

SCRIPT_STARTED = time.perf_counter()

import numpy as np
import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# Also allow `python examples/mnistAdd/mnistAdd.py` from any working directory.
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
from neurasp.cache import cached_compile_program
from neurasp.compiler import CompiledProgram, EnumerationOptions
from neurasp.device import autocast, memory_format
from neurasp.module import NeurASPModule, NeurASPProgram, semantic_loss

DATA_DIR = Path(__file__).resolve().parent / "data"


PROGRAM = """
img(i1). img(i2).
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
addition(i1,i2,N) :- digit(1,i1,A), digit(1,i2,B), N=A+B.
"""


# Output column n is sum n, as NLLLoss needs for label n.
OUTPUTS = [f"addition(i1,i2,{total})" for total in range(19)]
# The compile cache of PROGRAM (gitignored).
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def compile_mnist(
    options: EnumerationOptions = EnumerationOptions(),
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> tuple[CompiledProgram, str]:
    """Compile PROGRAM, enumerating its models as options say, through the compile
    cache entry mnistAdd in cache_dir if it is given (see
    neurasp.cache.cached_compile_program), and check its inputs. Returns the compiled
    program and where it came from: "cache" or "clingo".
    """
    compiled, source = cached_compile_program(
        PROGRAM, OUTPUTS, options, cache_dir=cache_dir, name="mnistAdd", rebuild=rebuild
    )
    if [atom.term for atom in compiled.neural_atoms] != ["i1", "i2"]:
        raise ValueError("Expected the two MNIST image inputs i1 and i2")
    return compiled, source


def build_module() -> NeurASPModule:
    """Compile PROGRAM into a module whose input columns are the digits 0..9 of image
    i1, then of image i2, and whose output column n is the probability of sum n.
    """
    return NeurASPModule(compile_mnist()[0].models)


class MNISTTuples(Dataset[tuple[torch.Tensor, int]]):
    """Rows of `examples` hold MNIST indices followed by the label; an item is the
    stacked images, of shape (images, 1, 28, 28), and the label.
    """

    def __init__(self, mnist: torchvision.datasets.MNIST, examples: Path) -> None:
        """Read the rows of the examples file and transform mnist's images once.

        torchvision's MNIST converts and normalizes an image on every access, which
        took 3.0 s of a 10–11 s GPU epoch; the transformed split is 188 MB for the
        60,000 training images.
        """
        self.images = normalized_images(mnist)
        self.rows = np.loadtxt(examples, dtype=np.int64, ndmin=2)

    def __len__(self) -> int:
        """Return the number of tuples."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return tuple index as its stacked images and its label."""
        *indices, label = self.rows[index].tolist()
        return self.images[indices], label


# The original example's normalization: the MNIST training set's mean and deviation.
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def mnist(train: bool) -> torchvision.datasets.MNIST:
    """Return the MNIST train or test split under data/, downloading it if needed."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    return torchvision.datasets.MNIST(
        root=str(DATA_DIR), train=train, download=True, transform=transform
    )


def normalized_images(mnist: torchvision.datasets.MNIST) -> torch.Tensor:
    """Every image of mnist as its transform gives it, (images, 1, 28, 28) float32,
    computed at once from the raw uint8 images: ToTensor scales them to [0, 1] and
    Normalize subtracts MNIST_MEAN and divides by MNIST_STD, both in float32.
    """
    images = mnist.data.unsqueeze(1).float().div(255)
    return images.sub(MNIST_MEAN).div(MNIST_STD)


@torch.no_grad()
def predict_digits(
    network: nn.Module, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """The argmax digit of each of images, (images, 1, 28, 28), on the CPU; network
    is put in evaluation mode and classifies 1000 images per call.
    """
    network.eval()
    return torch.cat(
        [network(batch.to(device)).argmax(-1).cpu() for batch in images.split(1000)]
    )


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Percentage of predictions equal to their labels."""
    return 100.0 * int((predictions == labels).sum()) / len(labels)


def downstream_accuracy(
    digits: torch.Tensor, mnist_test: torchvision.datasets.MNIST, test_file: Path
) -> tuple[float, int]:
    """Percentage of held-out tuples whose label is the sum of their images' predicted
    digits, and the number of tuples; digits holds the predicted digit of every image
    of mnist_test, as predict_digits returns.

    Applies addition to each image's predicted digit rather than taking the argmax of
    the marginal over all models, as the baseline evaluation does. Raises ValueError if
    test_file's labels do not match the MNIST test split.
    """
    rows = np.loadtxt(test_file, dtype=np.int64, ndmin=2)
    indices, labels = torch.from_numpy(rows[:, :-1]), torch.from_numpy(rows[:, -1])
    if not torch.equal(mnist_test.targets[indices].sum(1), labels):
        raise ValueError(f"{test_file} labels do not match the MNIST test split")
    return accuracy(digits[indices].sum(1), labels), len(labels)


def main() -> None:
    """Parse the command line, compile the program, train, evaluate and write the
    metrics.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_enumeration_arguments(parser)
    add_models_cache_arguments(parser, CACHE_DIR)
    add_training_arguments(parser, batch_size=16, epochs=1, learning_rate=0.001)
    args, device = parse_training_arguments(parser)
    options = enumeration_options(parser, args)
    output_dir = (
        None
        if args.compile_only
        else choose_output_dir(args.output_dir, "mnistAdd", device, args.batch_size)
    )
    seed_everything(args.seed, args.deterministic)

    with timer() as compilation:
        compiled, models_source = compile_mnist(
            options, CACHE_DIR if args.models_cache else None, args.rebuild_models
        )
        module = NeurASPModule(compiled.models)
    print(
        f"Models ({models_source}) and module: {compilation.seconds:.3f}s; "
        f"{module.model_count:,} stable models",
        flush=True,
    )
    if output_dir is None:  # --compile-only
        return
    network = Net().to(memory_format=memory_format(device, args.precision))
    # One digit network classifies both images, so it is optimized once.
    program = NeurASPProgram([(network, 2)], module).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

    mnist_train = mnist(train=True)
    train_dataset = MNISTTuples(mnist_train, DATA_DIR / "train_data.txt")
    # Not shuffled, as in the original NeurASP example.
    dataloader = data_loader(train_dataset, args, device)

    def step(batch: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        """One training step on a DataLoader batch; returns its loss and size."""
        images, labels = batch
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()
        # Only each example's observed label is needed for the loss, so evaluate just
        # the models of that label. The labels stay on the CPU: forward_single_output
        # reads the batch's number of models from them there and copies them over.
        with autocast(device, args.precision):
            predictions = program(list(images.unbind(1)), expected_outputs=labels)
            loss = semantic_loss(predictions)
        loss.backward()
        optimizer.step()
        return loss.detach(), len(labels)

    print(
        f"Start mnistAdd: {args.epochs} epoch(s), {len(train_dataset)} tuples, "
        f"batch size {args.batch_size}, {device}",
        flush=True,
    )
    with timer(device) as training:
        train_epochs(
            program,
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
        downstream, test_examples = downstream_accuracy(
            test_digits, mnist_test, DATA_DIR / "test_data.txt"
        )
    wall_seconds = time.perf_counter() - SCRIPT_STARTED

    metrics = {
        "task": "mnistAdd",
        **run_metadata(device, args),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_examples": len(train_dataset),
        # Held-out tuples, as in card_arithmetic; latent accuracy uses latent_test_images.
        "test_examples": test_examples,
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
        "notes": "Latent accuracy is digit classification on the MNIST test split (train: "
        "its first 1000 images). Downstream applies addition to argmax digits on "
        "held-out test tuples.",
    }
    write_metrics(output_dir, metrics)
    print_summary(output_dir, metrics)


if __name__ == "__main__":
    main()
