"""A disk cache of compiled programs: compile_program's result, keyed on what
determines it, so that a program's stable models are enumerated once and then read
back.

Each cache entry is one file, named by the caller (one name per program, such as a
task variant), holding the key it was written for. An entry whose key differs is
stale: it is ignored, and overwritten when the program is compiled again. Files are
read with torch.load(weights_only=True), so they hold only tensors and plain values.
"""

import hashlib
import inspect
import os
from pathlib import Path
from typing import Any, Sequence

import torch

import neurasp.compiler
import neurasp.generator
import neurasp.parser
from neurasp.compiler import CompiledProgram, EnumerationOptions, compile_program
from neurasp.parser import NeuralAtom

# Bumped when the file layout changes, which makes every existing entry stale.
CACHE_FORMAT = 1


def compile_key(program: str, outputs: Sequence[str]) -> str:
    """A hash of everything that determines compile_program(program, outputs): the
    program, the output atoms in order, the source of the code that compiles it (the
    parser, the compiler and the generator) and the cache format. Changing any of them
    gives another key. EnumerationOptions are not part of it: they change how the
    models are enumerated, never which.
    """
    sources = [str(CACHE_FORMAT), program, *outputs]
    sources += [
        inspect.getsource(code)
        for code in (neurasp.parser, neurasp.compiler, neurasp.generator)
    ]
    return hashlib.sha256("\0".join(sources).encode()).hexdigest()


def cache_path(cache_dir: Path, name: str) -> Path:
    """The file of the cache entry name in cache_dir."""
    return Path(cache_dir) / f"{name}.pt"


def load_compiled(path: Path, key: str) -> CompiledProgram | None:
    """The compiled program stored at path, or None if there is no file or it was
    written for another key (it is stale).
    """
    if not path.exists():
        return None
    stored = torch.load(path, weights_only=True)
    if stored.get("key") != key:
        return None
    blocks = stored["models"].split(stored["counts"])
    neural_atoms = tuple(
        NeuralAtom(
            raw=atom["raw"],
            network=atom["network"],
            output_count=atom["output_count"],
            term=atom["term"],
            domain=tuple(atom["domain"]),
        )
        for atom in stored["neural_atoms"]
    )
    return CompiledProgram(dict(enumerate(blocks)), neural_atoms)


def save_compiled(path: Path, key: str, compiled: CompiledProgram) -> None:
    """Store compiled at path for key: the models as one tensor of every output's
    models in output order with each output's count, and the neural atoms as plain
    values. The file is written under a temporary name and renamed, so an interrupted
    write never leaves a partial entry.
    """
    models = compiled.models
    neural_atoms: list[dict[str, Any]] = [
        {
            "raw": atom.raw,
            "network": atom.network,
            "output_count": atom.output_count,
            "term": atom.term,
            "domain": list(atom.domain),
        }
        for atom in compiled.neural_atoms
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "key": key,
            "models": torch.cat([models[column] for column in range(len(models))]),
            "counts": [len(models[column]) for column in range(len(models))],
            "neural_atoms": neural_atoms,
        },
        temporary,
    )
    os.replace(temporary, path)


def cached_compile_program(
    program: str,
    outputs: Sequence[str],
    options: EnumerationOptions = EnumerationOptions(),
    *,
    cache_dir: Path | None,
    name: str,
    rebuild: bool = False,
) -> tuple[CompiledProgram, str]:
    """compile_program(program, outputs, options) through the cache entry name in
    cache_dir.

    The compiled program is read from the entry when it is current (its key is
    compile_key's), and otherwise compiled and written to it; rebuild compiles and
    rewrites it even when it is current. With cache_dir None, nothing is read or
    written. Returns the compiled program and where it came from: "cache" or
    "clingo".
    """
    if cache_dir is None:
        return compile_program(program, outputs, options), "clingo"
    path = cache_path(cache_dir, name)
    key = compile_key(program, outputs)
    if not rebuild:
        cached = load_compiled(path, key)
        if cached is not None:
            return cached, "cache"
    compiled = compile_program(program, outputs, options)
    save_compiled(path, key, compiled)
    return compiled, "clingo"
