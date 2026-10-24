"""Train card arithmetic with the neurasp parser, generator and NeurASPProgram.

The NeurASP program is compiled once into a NeurASPModule (all 52**N stable models
enumerated by Clingo, sparse projection); training then runs in PyTorch alone. Four
cards need substantial time and memory. Run from the repository root:

    python -m examples.card_arithmetic.card_arithmetic   # sum_2, cpu, batch 16, 10 epochs
    python -m examples.card_arithmetic.card_arithmetic --variant sum_3 --device gpu
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

# Also allow `python examples/card_arithmetic/card_arithmetic.py` from any working directory.
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
from examples.card_arithmetic.card_dataset import (
    TASK_DIR,
    VARIANTS,
    CardArithmetic,
    add_card_arguments,
    card_images_to,
)
from examples.card_arithmetic.card_network import CardNet
from neurasp.cache import cached_compile_program
from neurasp.compiler import CompiledProgram, EnumerationOptions, compile_program
from neurasp.device import autocast, memory_format
from neurasp.module import NeurASPModule, NeurASPProgram, semantic_loss

# The compile cache of the card programs (gitignored).
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def card_values(variant: str) -> torch.Tensor:
    """Arithmetic value of each of the 52 classes, in the original h/c/s/d, 2..ace
    order: rank plus a suit offset for sum variants, rank times a suit factor for
    prodsum variants.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown card variant: {variant}")
    ranks = torch.arange(52) % 13 + 2
    suits = torch.arange(52) // 13
    if variant.startswith("sum_"):
        return ranks + torch.tensor([39, 13, 26, 0])[suits]
    return ranks * torch.tensor([4, 2, 3, 1])[suits]


def card_program(variant: str) -> str:
    """Return the variant's NeurASP program: card_facts.lp followed by
    card_<variant>.lp, with its neural slots shifted from 0..N-1 to the parser's 1..N.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown card variant: {variant}")
    count = int(variant.rsplit("_", 1)[1])
    rules = (TASK_DIR / f"card_{variant}.lp").read_text()
    replacements = {
        "sum(0,V) :- card_value(0,V).": "sum(1,V) :- card_value(1,V).",
        f"result(X) :- sum({count - 1},X).": f"result(X) :- sum({count},X).",
    }
    for original, shifted in replacements.items():
        if rules.count(original) != 1:
            raise ValueError(f"Unrecognized card rule, expected: {original}")
        rules = rules.replace(original, shifted, 1)
    return (TASK_DIR / "card_facts.lp").read_text() + "\n" + rules


def build_module(
    variant: str, options: EnumerationOptions = EnumerationOptions()
) -> tuple[NeurASPModule, list[int]]:
    """Compile the variant's program into a module whose input columns are the 52 card
    classes of slot 1, then of slot 2, and so on, and whose output column c is the
    probability of result_values[c]. Returns the module and result_values. options
    say how to enumerate the stable models (see compile_program).
    """
    models, result_values = build_card_models(variant, options)
    return NeurASPModule(models), result_values


def card_result_values(variant: str) -> list[int]:
    """Every possible result of the variant, in increasing order: the sums of the
    values (card_values) of N cards. Result value result_values[c] is output column c.
    """
    count = int(variant.rsplit("_", 1)[1])
    possible = {0}
    values = card_values(variant).tolist()
    for _ in range(count):
        possible = {left + right for left in possible for right in values}
    return sorted(possible)


def card_outputs(variant: str) -> list[str]:
    """The output atoms of the variant's program, result(v) for every possible result
    v in card_result_values order, so that output column c is result_values[c].
    """
    return [f"result({value})" for value in card_result_values(variant)]


def check_card_compiled(variant: str, compiled: CompiledProgram) -> None:
    """Check that the variant's compiled program has the one neural atom card(N,p),
    with classes 0..51 in neural output order. Raises ValueError otherwise.
    """
    count = int(variant.rsplit("_", 1)[1])
    if len(compiled.neural_atoms) != 1:
        raise ValueError("Expected exactly one neural atom, card(N,p)")
    atom = compiled.neural_atoms[0]
    if (atom.network, atom.term, atom.output_count, list(atom.domain)) != (
        "card",
        "p",
        count,
        [str(value) for value in range(52)],
    ):
        raise ValueError("Expected card(N,p) with classes 0..51 in neural output order")


def compile_card(
    variant: str, options: EnumerationOptions = EnumerationOptions()
) -> CompiledProgram:
    """Compile the variant's program (see compile_program, which enumerates the
    models as options say) and check it (check_card_compiled).

    The output atoms come from the task arithmetic (card_outputs), so compiling
    checks the program's results against it: compile_program rejects a listed result
    that is not grounded and a stable model whose result is not listed.
    """
    compiled = compile_program(card_program(variant), card_outputs(variant), options)
    check_card_compiled(variant, compiled)
    return compiled


def build_card_models(
    variant: str, options: EnumerationOptions = EnumerationOptions()
) -> tuple[dict[int, torch.Tensor], list[int]]:
    """Return the enumerated models (see compile_card) and the ordered semantic
    result values.
    """
    return compile_card(variant, options).models, card_result_values(variant)


def cached_card_models(
    variant: str,
    options: EnumerationOptions = EnumerationOptions(),
    *,
    cache: bool = True,
    rebuild: bool = False,
) -> tuple[dict[int, torch.Tensor], list[int], str]:
    """build_card_models(variant, options) through the compile cache: the entry
    card_<variant> in CACHE_DIR (see neurasp.cache.cached_compile_program).

    With cache, the models are read from the entry when it is current, and otherwise
    enumerated and written to it; rebuild enumerates and rewrites them even when it is
    current. Without cache, they are enumerated and the cache is neither read nor
    written. The compiled program is checked either way. Returns the models, the
    result values and where the models came from: "cache" or "clingo".
    """
    compiled, source = cached_compile_program(
        card_program(variant),
        card_outputs(variant),
        options,
        cache_dir=CACHE_DIR if cache else None,
        name=f"card_{variant}",
        rebuild=rebuild,
    )
    check_card_compiled(variant, compiled)
    return compiled.models, card_result_values(variant), source


def train_step(
    program: NeurASPProgram,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    columns: torch.Tensor,
    device: torch.device,
    precision: str = "ieee",
) -> torch.Tensor:
    """One optimizer step on a batch of card tuples, images of shape (batch, cards, 3,
    H, W), and the output columns of their results, on the CPU (the batch's number of
    models is read from them there); returns the
    detached mean loss, left
    on the device. A function rather than inline in main() so that
    bench_card_step.py times exactly this step.

    Nothing here reads a device value back, so on CUDA the step never makes the CPU wait
    for the GPU.
    """
    images = card_images_to(images, device)
    optimizer.zero_grad()
    # Only each example's observed result is needed for the loss, so evaluate just
    # the models of that result instead of all 52**count of them. The columns stay on
    # the CPU: forward_single_output reads the batch's number of models from them
    # there, without a sync, and copies them to the device.
    with autocast(device, precision):
        predictions = program(list(images.unbind(1)), expected_outputs=columns)
        loss = semantic_loss(predictions)
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def evaluate(
    network: nn.Module,
    dataloader: Iterable[Sequence[torch.Tensor]],
    values: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Downstream and latent accuracy, in percent, of network on dataloader's batches of
    (images, columns, results, cards); the columns are not used.

    As in the baseline, downstream accuracy applies the arithmetic to each card's argmax
    class and compares it with the result, which differs from taking the argmax after
    marginalizing all assignments. Latent accuracy counts individual cards, not
    completely correct tuples.
    """
    network.eval()
    values = values.to(device)
    downstream_correct = latent_correct = example_count = card_count = 0
    # The results, not the columns: the arithmetic on the predicted cards gives values.
    for images, _, results, latent_labels in dataloader:
        images = card_images_to(images, device)
        results, latent_labels = results.to(device), latent_labels.to(device)
        # All cards of all tuples in one call, tuple by tuple: (batch * cards, 52).
        predictions = (
            network(images.flatten(0, 1))
            .reshape(len(images), images.shape[1], 52)
            .argmax(-1)
        )
        downstream_correct += int((values[predictions].sum(1) == results).sum())
        latent_correct += int((predictions == latent_labels).sum())
        example_count += len(results)
        card_count += latent_labels.numel()
    if example_count == 0:
        raise ValueError("Cannot evaluate an empty card dataset")
    return (
        100.0 * downstream_correct / example_count,
        100.0 * latent_correct / card_count,
    )


def main() -> None:
    """Parse the command line, compile the variant, train, evaluate and write the
    metrics.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_card_arguments(parser)
    add_enumeration_arguments(parser)
    add_models_cache_arguments(parser, CACHE_DIR)
    add_training_arguments(parser, batch_size=16, epochs=10, learning_rate=0.0001)
    args, device = parse_training_arguments(parser)
    options = enumeration_options(parser, args)
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be nonnegative")
    task = f"card_{args.variant}"
    count = int(args.variant.rsplit("_", 1)[1])
    output_dir = (
        None
        if args.compile_only
        else choose_output_dir(args.output_dir, task, device, args.batch_size)
    )
    seed_everything(args.seed, args.deterministic)

    if count == 4:
        print(
            "Four-card compilation and batched autograd can require several GB of RAM.",
            flush=True,
        )
    with timer() as compilation:
        models, result_values, models_source = cached_card_models(
            args.variant,
            options,
            cache=args.models_cache,
            rebuild=args.rebuild_models,
        )
        module = NeurASPModule(models)
    print(
        f"Models ({models_source}) and module: {compilation.seconds:.3f}s; "
        f"{module.model_count:,} stable models",
        flush=True,
    )
    if output_dir is None:  # --compile-only
        return
    network = CardNet().to(memory_format=memory_format(device, args.precision))
    # One card network classifies every card, so it is optimized once.
    program = NeurASPProgram([(network, count)], module).to(device)
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
        loss = train_step(program, optimizer, images, columns, device, args.precision)
        return loss, len(columns)

    print(
        f"Start {task}: {args.epochs} epoch(s), {len(train_dataset)} tuples, "
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
            log_every=100,
        )

    # Built before timing, so evaluation_seconds is the evaluation alone, as for mnistAdd.
    # Latent train accuracy uses the first training tuples, as mnistAdd uses the first
    # 1000 training images; all of them would add a full pass of JPEG decoding.
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
        "train_examples": len(train_dataset),
        "test_examples": len(datasets["test"]),
        "cards_per_example": count,
        "downstream_accuracy_pct": downstream_accuracy,
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
        "result_values": result_values,
        **environment_metadata(),
        "notes": "Final weights; downstream uses arithmetic on each card's argmax class. "
        "Latent train accuracy uses the first 1000 training tuples. "
        "Training uses mean semantic NLL.",
    }
    write_metrics(output_dir, metrics)
    print_summary(output_dir, metrics)


if __name__ == "__main__":
    main()
