"""Benchmark of compiling a card program: enumerating its stable models with Clingo
under several EnumerationOptions.

Each configuration runs compile_card (parse, ground, enumerate, group by result;
not the module, nor training) and is timed on its own. The models of every
configuration are checked to be those of the first: the same models for each result,
whatever their order. Then writing the first configuration's compiled program to the
compile cache (neurasp.cache) and reading them back (cold, from disk, and warm, from the page cache) is timed
against its enumeration (--no-cache skips it). Run from the repository root:

    python -m examples.card_arithmetic.bench_card_compile   # sum_4, default grid
    python -m examples.card_arithmetic.bench_card_compile --variant sum_3 \\
        --config 1,1,compete,auto --config 8,1,compete,record

A configuration is PROCESSES,THREADS,PARALLEL_MODE,ENUM_MODE (see EnumerationOptions).
"""

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

# Also allow `python examples/card_arithmetic/bench_card_compile.py` from anywhere.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.card_arithmetic.card_arithmetic import (
    card_outputs,
    card_program,
    compile_card,
)
from examples.card_arithmetic.card_dataset import VARIANTS
from neurasp.cache import cache_path, compile_key, load_compiled, save_compiled
from neurasp.compiler import CompiledProgram, EnumerationOptions

# The default grid: processes alone, then Clingo's threads, and record enumeration
# (on sum_4 it takes far longer than backtracking in a single process).
DEFAULT_CONFIGS = [
    "1,1,compete,auto",
    "2,1,compete,auto",
    "4,1,compete,auto",
    "8,1,compete,auto",
    "12,1,compete,auto",
    "16,1,compete,auto",
    "20,1,compete,auto",
    "1,2,split,auto",
    "1,4,compete,auto",
    "8,2,split,auto",
    "16,1,compete,record",
]


def parse_config(text: str) -> EnumerationOptions:
    """Parse PROCESSES,THREADS,PARALLEL_MODE,ENUM_MODE into checked options. Raises
    argparse.ArgumentTypeError on a malformed or invalid configuration.
    """
    try:
        processes, threads, parallel_mode, enum_mode = text.split(",")
        options = EnumerationOptions(
            threads=int(threads),
            parallel_mode=parallel_mode,
            enum_mode=enum_mode,
            processes=int(processes),
        )
        options.check()
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid configuration {text}: {error}")
    return options


def canonical(models: dict[int, torch.Tensor]) -> list[np.ndarray]:
    """Each result's models as the lexicographically sorted rows of its matrix, so
    that enumerations in different orders compare equal.
    """
    return [np.unique(models[column].numpy(), axis=0) for column in sorted(models)]


def main() -> None:
    """Parse the command line, compile the variant under each configuration and
    print the times.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=VARIANTS, default="sum_4")
    parser.add_argument(
        "--config",
        type=parse_config,
        action="append",
        help="PROCESSES,THREADS,PARALLEL_MODE,ENUM_MODE; repeat for several "
        "(default: a grid of processes, threads and enumeration modes)",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also time writing the first configuration's models to the models cache "
        "and reading them back (default; in a temporary directory)",
    )
    args = parser.parse_args()
    configs = args.config or [parse_config(text) for text in DEFAULT_CONFIGS]

    reference: list[np.ndarray] | None = None
    baseline: float | None = None
    print(f"card_{args.variant}: compilation (parse, ground, enumerate, group)")
    print(
        f"{'processes':>9} {'threads':>7} {'mode':>8} {'enum':>7} "
        f"{'seconds':>8} {'speedup':>8} {'models':>10}",
        flush=True,
    )
    first: CompiledProgram | None = None
    for options in configs:
        started = time.perf_counter()
        compiled = compile_card(args.variant, options)
        seconds = time.perf_counter() - started
        baseline = baseline or seconds
        first = first or compiled
        models = compiled.models
        count = sum(len(block) for block in models.values())
        rows = canonical(models)
        if reference is None:
            reference = rows
        elif len(rows) != len(reference) or not all(
            np.array_equal(left, right) for left, right in zip(rows, reference)
        ):
            raise SystemExit(f"{options} enumerated different models")
        print(
            f"{options.processes:>9} {options.threads:>7} "
            f"{options.parallel_mode:>8} {options.enum_mode:>7} "
            f"{seconds:>8.2f} {baseline / seconds:>7.2f}x {count:>10,}",
            flush=True,
        )
    assert first is not None and reference is not None and baseline is not None
    if args.cache:
        time_cache(args.variant, first, reference, baseline)


def time_cache(
    variant: str,
    compiled: CompiledProgram,
    reference: list[np.ndarray],
    baseline: float,
) -> None:
    """Time writing compiled to the compile cache (in a temporary directory) and
    reading it back, cold (the file evicted from the page cache, so it is read from
    disk) and warm (from memory), against baseline, the first configuration's seconds.
    A read includes computing the key, as cached_compile_program does. Exits if the
    cache gives other models or neural atoms.
    """
    program, outputs = card_program(variant), card_outputs(variant)
    with tempfile.TemporaryDirectory() as directory:
        path = cache_path(Path(directory), f"card_{variant}")
        started = time.perf_counter()
        save_compiled(path, compile_key(program, outputs), compiled)
        save_seconds = time.perf_counter() - started
        size = path.stat().st_size
        print(f"cache file: {size / 2**20:,.1f} MiB; write {save_seconds:.2f}s")
        for label in ("cold", "warm"):
            if label == "cold":
                # Write the file out, then drop its pages from the page cache.
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            started = time.perf_counter()
            cached = load_compiled(path, compile_key(program, outputs))
            seconds = time.perf_counter() - started
            if cached is None or cached.neural_atoms != compiled.neural_atoms:
                raise SystemExit("the cache did not return the compiled program")
            rows = canonical(cached.models)
            if not all(np.array_equal(a, b) for a, b in zip(rows, reference)):
                raise SystemExit("the cache returned different models")
            print(
                f"cache read ({label}): {seconds:.3f}s, "
                f"{baseline / seconds:,.0f}x faster than the first configuration",
                flush=True,
            )


if __name__ == "__main__":
    main()
