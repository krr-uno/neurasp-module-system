"""Shared plumbing for the training examples: command line, seeding, timing, the
epoch loop and the metrics files. Task-specific compilation, training steps and
evaluation stay in each example."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from neurasp.compiler import ENUM_MODES, PARALLEL_MODES, EnumerationOptions
from neurasp.device import PRECISIONS, device_metadata, resolve_device, synchronize

ROOT = Path(__file__).resolve().parents[1]


def add_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    batch_size: int,
    epochs: int,
    learning_rate: float,
) -> None:
    """Add the options every example takes, with the example's own defaults."""
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--learning-rate", "--lr", type=float, default=learning_rate)
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=1537081416)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use only deterministic GPU algorithms, so the same seed gives the same "
        "result (slower; fails on an operation that has none)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="ieee",
        help="CUDA arithmetic: ieee full float32 (default); tf32 for matrix products "
        "and convolutions; bf16 also trains under bfloat16 autocast (evaluation stays "
        "in full precision), with convolution weights in the channels_last layout",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="processes that load (decode) batches; 0 (the default) loads them in the "
        "main process, which on a GPU already overlaps with training because the step "
        "never waits for the GPU",
    )


def add_enumeration_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options that say how Clingo enumerates the stable models (see
    EnumerationOptions); read them with enumeration_options.
    """
    parser.add_argument(
        "--compile-processes",
        type=int,
        default=1,
        help="processes that enumerate the stable models, each for some values of the "
        "first neural input (default 1)",
    )
    parser.add_argument(
        "--clingo-threads",
        type=int,
        default=1,
        help="Clingo solver threads in each process (default 1)",
    )
    parser.add_argument(
        "--clingo-parallel-mode", choices=PARALLEL_MODES, default="compete"
    )
    parser.add_argument("--clingo-enum-mode", choices=ENUM_MODES, default="auto")


def enumeration_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> EnumerationOptions:
    """The EnumerationOptions from add_enumeration_arguments' options. Exits through
    parser.error on invalid values.
    """
    options = EnumerationOptions(
        threads=args.clingo_threads,
        parallel_mode=args.clingo_parallel_mode,
        enum_mode=args.clingo_enum_mode,
        processes=args.compile_processes,
    )
    try:
        options.check()
    except ValueError as error:
        parser.error(str(error))
    return options


def add_models_cache_arguments(
    parser: argparse.ArgumentParser, cache_dir: Path
) -> None:
    """Add the options of the compile cache in cache_dir (see neurasp.cache):
    --models-cache/--no-models-cache, --rebuild-models and --compile-only.
    """
    shown = cache_dir.relative_to(ROOT) if cache_dir.is_relative_to(ROOT) else cache_dir
    parser.add_argument(
        "--models-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"read the compiled program from {shown}/ when it is current, otherwise "
        "compile and write it (default); --no-models-cache always compiles and leaves "
        "the cache alone",
    )
    parser.add_argument(
        "--rebuild-models",
        action="store_true",
        help="compile even if the cache is current, and rewrite it",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="obtain the stable models as the other cache options say (so, by "
        "default, compile and cache them unless the cache is current), then exit "
        "without training or writing metrics",
    )


def parse_training_arguments(
    parser: argparse.ArgumentParser,
) -> tuple[argparse.Namespace, torch.device]:
    """Parse and check the options from add_training_arguments; return them and the
    resolved device. Exits through parser.error on invalid values.
    """
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or not 0 <= args.seed < 2**32:
        parser.error(
            "batch size and epochs must be positive; seed must be in 0..2**32-1"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive")
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    try:
        device = resolve_device(args.device, args.precision)
    except ValueError as error:
        parser.error(str(error))
    if device.type == "cuda" and "OMP_NUM_THREADS" not in os.environ:
        limit_cpu_threads()
    return args, device


def limit_cpu_threads() -> None:
    """Use at most half the CPU cores for PyTorch's intra-op threads.

    On a GPU the main process runs only small CPU operations (collating and pinning
    batches), but by default each one starts a thread per core. With every core
    taken, the threads spin waiting for each other while the main thread, which
    launches the GPU work, waits for a core: on a 20-core Intel Core Ultra 7 265 (8
    performance and 12 efficiency cores), a card sum_3 epoch took 46 s with 20 threads,
    24 s with 19 and 13 s with 16 or fewer. Setting OMP_NUM_THREADS overrides this.
    """
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python's, NumPy's and PyTorch's random number generators.

    Seeding alone does not make GPU runs repeatable: some cuDNN and CUDA kernels are
    nondeterministic. With deterministic, PyTorch uses only deterministic algorithms
    (raising an error on an operation that has none) and cuDNN does not autotune.
    Call it before any CUDA work: cuBLAS reads its workspace setting when it starts.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def data_loader(
    dataset: Dataset[Any],
    args: argparse.Namespace,
    device: torch.device,
    *,
    shuffle: bool = False,
    batch_size: int | None = None,
) -> DataLoader[Any]:
    """A DataLoader with the run's batch size (or batch_size) and --workers worker
    processes, kept alive across epochs, so batches are loaded while the device
    trains. On CUDA, pinned host memory lets the copies to the GPU run asynchronously.

    The workers are forked, not started by Python's default forkserver: they start at
    once instead of re-importing the script, and share the dataset's memory (such as
    preloaded images) instead of each receiving a copy. They never use CUDA, so
    forking after CUDA has started is safe.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        multiprocessing_context="fork" if args.workers > 0 else None,
        pin_memory=device.type == "cuda",
    )


def choose_output_dir(
    output_dir: Path | None, task: str, device: torch.device, batch_size: int
) -> Path:
    """Return the run's output directory, output_dir or by default a new timestamped
    directory under benchmark_results/, without creating it: write_metrics does, so a
    run that fails leaves nothing behind. Raises FileExistsError if it already exists,
    so runs never overwrite each other and a taken directory fails before training.
    """
    if output_dir is None:
        output_dir = (
            ROOT
            / "benchmark_results"
            / (
                f"pipeline_{task}_{'gpu' if device.type == 'cuda' else 'cpu'}"
                f"_batch{batch_size}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
        )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    return output_dir


class Elapsed:
    """Seconds measured by timer(); set when the timed block ends."""

    seconds: float = 0.0


@contextmanager
def timer(device: torch.device | None = None) -> Iterator[Elapsed]:
    """Time the block into the yielded object's .seconds. With a device, wait for it at
    both ends so that asynchronous GPU work is included.
    """
    elapsed = Elapsed()
    if device is not None:
        synchronize(device)
    started = time.perf_counter()
    yield elapsed
    if device is not None:
        synchronize(device)
    elapsed.seconds = time.perf_counter() - started


def train_epochs(
    model: nn.Module,
    batches: DataLoader[Any],
    step: Callable[[Any], tuple[Tensor, int]],
    *,
    epochs: int,
    examples: int,
    device: torch.device,
    started: float,
    log_every: int,
) -> None:
    """Run step on every batch for the given epochs, printing progress every log_every
    batches and the mean loss per example after each epoch.

    step(batch) does one optimizer step and returns the batch's detached mean loss and
    its number of examples. The loss is summed on the device: loss.item() would force a
    GPU->CPU sync every batch. started is the script's start time, for the progress
    lines.
    """
    for epoch in range(epochs):
        model.train()
        running_loss = torch.zeros((), device=device)
        for index, batch in enumerate(batches, 1):
            loss, size = step(batch)
            running_loss += loss * size
            if index % log_every == 0:
                print(
                    f"Epoch {epoch + 1}, batch {index}/{len(batches)}. "
                    f"Time from start: {time.perf_counter() - started:.2f}s",
                    flush=True,
                )
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss.item() / examples:.4f}",
            flush=True,
        )


def run_metadata(device: torch.device, args: argparse.Namespace) -> dict[str, object]:
    """The leading metrics fields of a completed run, from its device and options."""
    return {
        "run": 1,
        "status": "ok",
        **device_metadata(device, args.precision),
        "deterministic": args.deterministic,
    }


def environment_metadata() -> dict[str, object]:
    """The trailing metrics fields describing the software environment."""
    return {
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "python_version": sys.version,
    }


def write_metrics(output_dir: Path, metrics: dict[str, object]) -> None:
    """Create output_dir (it must not exist yet; see choose_output_dir) and write
    metrics to it as metrics.json and as a one-row runs.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    with (output_dir / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def print_summary(output_dir: Path, metrics: dict[str, object]) -> None:
    """Print the final lines of a run from its metrics: downstream and latent accuracy
    (skipped if the run has none, as for purely neural baselines), times and where the
    results are.
    """
    print(
        f"Downstream test accuracy: {metrics['downstream_accuracy_pct']:.2f}%",
        flush=True,
    )
    if metrics["latent_accuracy_pct"] is not None:
        print(
            f"Latent accuracy: train {metrics['latent_train_accuracy_pct']:.2f}%, "
            f"test {metrics['latent_accuracy_pct']:.2f}%",
            flush=True,
        )
    print(
        f"Training: {metrics['training_seconds']:.2f}s; "
        f"total: {metrics['wall_seconds']:.2f}s",
        flush=True,
    )
    print(f"Results: {output_dir}", flush=True)
