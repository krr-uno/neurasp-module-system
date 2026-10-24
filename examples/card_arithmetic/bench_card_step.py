"""Microbenchmark of card_arithmetic training steps, for fast optimization iterations.

Times a few batches of the exact step that card_arithmetic.py trains with
(card_arithmetic.train_step), so changes to NeurASPModule, CardNet or the step itself
show up here first. To keep the time between iterations short:

- The enumerated models are cached in examples/card_arithmetic/.cache/ (gitignored).
  Only Clingo's enumeration is cached, not the built NeurASPModule, so changes to the
  module are always measured. The cache, shared with card_arithmetic.py, is rebuilt
  when the card program or the code that compiles it changes (see
  card_arithmetic.models_cache_key); --rebuild forces it.
- The batches are decoded from JPEG once, before timing, and kept in pinned host
  memory. Each step still copies its batch to the device, as in training. The decode
  time is printed separately: it is part of a real epoch but not of the step.

After the steps, NeurASPModule and CardNet are also timed on their own, with random
card probabilities in place of CardNet's output and a random gradient in place of
NeurASPModule's, to show how much of the step each accounts for. The profile table
cannot separate the two: some operators, such as clamp_min and sum, run in both.

Run from the repository root:

    python -m examples.card_arithmetic.bench_card_step   # sum_3, cuda, batch 16
    python -m examples.card_arithmetic.bench_card_step --profile   # plus a per-op table
    python -m examples.card_arithmetic.bench_card_step --trace trace.json   # Chrome trace
"""

import argparse
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Sequence

SCRIPT_STARTED = time.perf_counter()

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from examples.card_arithmetic.card_dataset import (
    ROOT,
    VARIANTS,
    CardArithmetic,
    card_images_to,
)
from examples.card_arithmetic.card_network import CardNet
from neurasp.device import (
    PRECISIONS,
    autocast,
    memory_format,
    resolve_device,
    synchronize,
)
from examples.card_arithmetic.card_arithmetic import (
    cached_card_models,
    train_step,
)
from neurasp.module import NeurASPModule, NeurASPProgram, semantic_loss


def load_batches(
    variant: str,
    image_dir: Path,
    result_values: list[int],
    batch_size: int,
    count: int,
    pin: bool,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The first count training batches, in file order, as (images, columns) pairs."""
    dataset = CardArithmetic(image_dir, variant, "train", result_values)
    if len(dataset) < batch_size * count:
        raise ValueError(f"{variant} has only {len(dataset)} training tuples")
    loader = DataLoader(
        Subset(dataset, range(batch_size * count)),
        batch_size=batch_size,
        pin_memory=pin,
    )
    return [(images, columns) for images, columns, _, _ in loader]


def time_neurasp(
    module: NeurASPModule,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    device: torch.device,
) -> float:
    """Milliseconds per step of NeurASPModule's forward and backward alone, as
    train_step runs them, over the batches after the first warmup ones.

    The batches' real result columns pick the models, so the work matches training; the
    card probabilities are random, since their values do not change the work done.
    """
    cards = batches[0][0].shape[1]
    inputs = []
    for images, columns in batches:
        probabilities = torch.rand(
            len(columns), 52 * cards, device=device
        ).requires_grad_()
        inputs.append((probabilities, columns.to(device), module.model_total(columns)))

    def step(probabilities: torch.Tensor, columns: torch.Tensor, total: int) -> None:
        """One forward and backward pass of the module on precomputed probabilities,
        with the batch's total model count, as train_step runs it.
        """
        probabilities.grad = None
        predictions = module.forward_single_output(probabilities, columns, total)
        semantic_loss(predictions).backward()

    return _time_steps(step, inputs, warmup, device)


def time_network(
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    device: torch.device,
    precision: str = "ieee",
) -> float:
    """Milliseconds per step of the rest of train_step with NeurASPModule left out: the
    image copy, zero_grad, CardNet's forward and backward, and the optimizer step, over
    the batches after the first warmup ones.

    A random gradient stands in for the one NeurASPModule sends back to CardNet's
    output; its values do not change the work done. This updates network's weights.
    """
    inputs = []
    for images, columns in batches:
        gradient = torch.rand(len(columns) * images.shape[1], 52, device=device)
        inputs.append((images, gradient))

    def step(images: torch.Tensor, gradient: torch.Tensor) -> None:
        """One training step of CardNet alone, with gradient in place of
        NeurASPModule's.
        """
        images = card_images_to(images, device)
        optimizer.zero_grad()
        # As NeurASPProgram calls it: every card slot stacked along the batch dimension.
        with autocast(device, precision):
            probabilities = network(torch.cat(images.unbind(1)))
        probabilities.backward(gradient)
        optimizer.step()

    return _time_steps(step, inputs, warmup, device)


def _time_steps(
    step: Callable[..., None],
    inputs: Sequence[tuple[Any, ...]],
    warmup: int,
    device: torch.device,
) -> float:
    """Milliseconds per call of step(*input) over inputs[warmup:], after untimed calls
    on inputs[:warmup]. On CUDA, the calls are timed with events and not synced between
    them, so they overlap as in training.
    """
    for input in inputs[:warmup]:
        step(*input)
    timed = inputs[warmup:]
    synchronize(device)
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
    started = time.perf_counter()
    for input in timed:
        step(*input)
    if device.type == "cuda":
        end.record()
        synchronize(device)
        return start.elapsed_time(end) / len(timed)
    return 1000 * (time.perf_counter() - started) / len(timed)


def main() -> None:
    """Parse the command line, time the training steps and print the measurements."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=VARIANTS, default="sum_3")
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=10, help="timed batches")
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="untimed batches first (CUDA context, cuDNN autotuning, allocator)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="ieee",
        help="CUDA arithmetic, as in the training scripts: ieee, tf32 or bf16",
    )
    parser.add_argument(
        "--image-dir", type=Path, default=ROOT / "examples/card_arithmetic/images"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="ignore the cached models"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print the operators with the most device time over the timed batches",
    )
    parser.add_argument(
        "--trace", type=Path, help="write a Chrome trace of the timed batches"
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.batches < 1 or args.warmup < 0:
        parser.error(
            "batch size and batches must be positive; warmup must be nonnegative"
        )
    try:
        device = resolve_device(args.device, args.precision)
    except ValueError as error:
        parser.error(str(error))
    torch.manual_seed(args.seed)

    started = time.perf_counter()
    models, result_values, source = cached_card_models(
        args.variant, rebuild=args.rebuild
    )
    models_seconds = time.perf_counter() - started
    started = time.perf_counter()
    module = NeurASPModule(models).to(device)
    module_seconds = time.perf_counter() - started
    started = time.perf_counter()
    batches = load_batches(
        args.variant,
        args.image_dir,
        result_values,
        args.batch_size,
        args.warmup + args.batches,
        pin=device.type == "cuda",
    )
    decode_seconds = time.perf_counter() - started
    network = CardNet().to(memory_format=memory_format(device, args.precision))
    cards = batches[0][0].shape[1]
    program = NeurASPProgram([(network, cards)], module).to(device)
    program.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    print(
        f"card_{args.variant}: {module.model_count:,} models, up to "
        f"{module.max_model_count:,} per result; batch {args.batch_size}, {device}"
    )
    print(
        f"setup: models {models_seconds:.2f}s ({source}), module {module_seconds:.2f}s, "
        f"decode {len(batches)} batches {decode_seconds:.2f}s"
    )

    for images, columns in batches[: args.warmup]:
        train_step(program, optimizer, images, columns, device, args.precision)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    profiler = None
    if args.profile or args.trace:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities)
        profiler.start()
    # CUDA events time each step on the device's stream without syncing between steps,
    # so the steps overlap as they do in training. On CPU every op is synchronous anyway.
    timed = batches[args.warmup :]
    if device.type == "cuda":
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(timed) + 1)]
        events[0].record()
    cpu_marks = [time.perf_counter()]
    losses = []
    for step, (images, columns) in enumerate(timed, 1):
        losses.append(
            train_step(program, optimizer, images, columns, device, args.precision)
        )
        if device.type == "cuda":
            events[step].record()
        cpu_marks.append(time.perf_counter())
    synchronize(device)
    wall = time.perf_counter() - cpu_marks[0]
    if profiler:
        profiler.stop()

    if device.type == "cuda":
        step_ms = [start.elapsed_time(end) for start, end in zip(events, events[1:])]
    else:
        step_ms = [1000 * (end - start) for start, end in zip(cpu_marks, cpu_marks[1:])]
    # Time for the CPU to queue a step. If it is close to the step time, the GPU is
    # waiting on Python/launch overhead rather than on its own work.
    launch_ms = [1000 * (end - start) for start, end in zip(cpu_marks, cpu_marks[1:])]
    mean_ms = 1000 * wall / len(timed)
    print(
        f"step ms: mean {mean_ms:.1f}, median {statistics.median(step_ms):.1f}, "
        f"min {min(step_ms):.1f}, max {max(step_ms):.1f}  "
        f"({args.batch_size * len(timed) / wall:.0f} examples/s)"
    )
    print(f"per step: " + " ".join(f"{ms:.1f}" for ms in step_ms))
    print(f"CPU launch ms: median {statistics.median(launch_ms):.1f}")
    if device.type == "cuda":
        print(
            f"peak allocated: {torch.cuda.max_memory_allocated(device) / 2**20:,.0f} MiB"
        )
    train_size = len(
        CardArithmetic(args.image_dir, args.variant, "train", result_values)
    )
    epoch_steps = -(-train_size // args.batch_size)
    print(
        f"epoch estimate ({epoch_steps} steps): {epoch_steps * mean_ms / 1000:.1f}s compute "
        f"+ {epoch_steps * decode_seconds / len(batches):.1f}s single-process decode"
    )
    print(f"final loss {losses[-1].item():.4f}")
    # After the profiler has stopped and peak memory was read, so neither includes them.
    # time_network trains the network further, so this also comes after the final loss.
    neurasp_ms = time_neurasp(module, batches, args.warmup, device)
    network_ms = time_network(
        network, optimizer, batches, args.warmup, device, args.precision
    )
    print(
        f"alone, ms/step (% of the mean step): "
        f"NeurASP forward+backward {neurasp_ms:.2f} ({100 * neurasp_ms / mean_ms:.2f}%), "
        f"CardNet forward+backward+optimizer {network_ms:.1f} ({100 * network_ms / mean_ms:.1f}%), "
        f"unaccounted {mean_ms - neurasp_ms - network_ms:.1f}"
    )
    print(f"total {time.perf_counter() - SCRIPT_STARTED:.1f}s")

    if profiler:
        sort = (
            "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        )
        if args.profile:
            print(profiler.key_averages().table(sort_by=sort, row_limit=20))
        if args.trace:
            profiler.export_chrome_trace(str(args.trace))
            print(f"trace: {args.trace}")


if __name__ == "__main__":
    main()
