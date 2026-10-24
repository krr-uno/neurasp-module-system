"""Compile a NeurASP program into the stable models a NeurASPModule takes: parse,
ground with Clingo, describe the neural inputs and outputs, enumerate the models."""

from concurrent.futures import ProcessPoolExecutor
import itertools
import multiprocessing
import re
from typing import NamedTuple, Sequence

import numpy as np
import torch
from clingo.control import Control
from clingo.core import Library

from clingo.symbol import parse_term as parse_symbol_term

from neurasp.generator import (
    ENUM_MODES,
    InputDescription,
    ModuleGenerator,
    combination_count,
    group_models,
    symbol_id_from_string,
    symbol_signature,
)
from neurasp.parser import NeuralAtom, format_network_atom, parse_neurasp_program


class CompiledProgram(NamedTuple):
    """The result of compile_program: the stable models and the neural atoms, which
    give the order of the module's input columns.
    """

    # Output column -> a (models, inputs) matrix of the stable models' input-column
    # positions; the argument of NeurASPModule.
    models: dict[int, torch.Tensor]
    # The neural atoms, in input-column order.
    neural_atoms: tuple[NeuralAtom, ...]


def _input_order(atom: NeuralAtom) -> tuple[str, list[tuple[int, int | str]]]:
    """Sort key of a neural atom in the input columns: by network, so each network's
    inputs are consecutive as NeurASPProgram needs, then by term in natural order,
    where runs of digits compare as numbers, so i2 comes before i10.
    """
    natural = [
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", atom.term)
        if part
    ]
    return atom.network, natural


# Clingo's parallel modes: threads that compete on the whole search space, or that
# split it between them.
PARALLEL_MODES = ("compete", "split")


class EnumerationOptions(NamedTuple):
    """How compile_program enumerates the stable models. The options change only how
    fast the models are enumerated, not which: the models are the same, though their
    order within an output may differ.

    threads is the number of Clingo solver threads and parallel_mode how they share
    the search (PARALLEL_MODES); enum_mode is Clingo's enumeration mode (ENUM_MODES).
    With processes above 1, the enumeration is split by the value of the first input:
    one task per atom of the first input group, each grounding the program again
    (cheap) and enumerating its models under the assumption that the atom is true, run
    by that many processes. Unlike Clingo's threads, this also runs the reading
    of the models, done in Python, in parallel.
    """

    threads: int = 1
    parallel_mode: str = "compete"
    enum_mode: str = "auto"
    processes: int = 1

    def check(self) -> None:
        """Raise ValueError for threads or processes below 1 or an unknown mode."""
        if self.threads < 1 or self.processes < 1:
            raise ValueError("threads and processes must be at least 1")
        if self.parallel_mode not in PARALLEL_MODES:
            raise ValueError(
                f"parallel_mode must be one of {', '.join(PARALLEL_MODES)}"
            )
        if self.enum_mode not in ENUM_MODES:
            raise ValueError(f"enum_mode must be one of {', '.join(ENUM_MODES)}")


def compile_program(
    program: str,
    outputs: Sequence[str],
    options: EnumerationOptions = EnumerationOptions(),
) -> CompiledProgram:
    """Compile program, whose outputs are the ground atoms outputs, such as
    ["addition(i1,i2,0)", ..., "addition(i1,i2,18)"], enumerating its stable models
    as options say.

    Input columns: the neural atoms sorted by network and then by term in natural order
    (see _input_order), each atom's output slots 1..e in order, each slot's domain
    values in order; so a network's outputs are concatenated in that order. Output
    columns: outputs, in the given order, so output column n is the probability of
    outputs[n]. Raises ValueError if an output atom cannot be derived from the program
    (it is not in the grounded program), or if a stable model has no output atom or
    more than one; see ModuleGenerator.generate_models for the other requirements on
    the program, and EnumerationOptions.check for invalid options.
    """
    options.check()
    parsed = parse_neurasp_program(program)
    neural_atoms = tuple(sorted(parsed.neural_atoms, key=_input_order))
    inputs: list[list[str]] = []
    for atom in neural_atoms:
        # Each output slot of each neural atom is one categorical input.
        inputs.extend(
            [
                format_network_atom(atom.network, slot, atom.term, value)
                for value in atom.domain
            ]
            for slot in range(1, atom.output_count + 1)
        )
    task = _Task(
        parsed.standard_asp_program,
        inputs,
        list(outputs),
        options.threads,
        options.parallel_mode,
    )
    grounded = task.ground()
    if options.processes == 1:
        models = grounded.generator.generate_models(
            grounded.description, grounded.output_literals, options.enum_mode
        )
        return CompiledProgram(models, neural_atoms)

    # The processes are started by a fork server, not forked from this process, which
    # may have threads (PyTorch's, for one) that forking would copy in an unknown
    # state; starting the server costs about a second, once per Python process.
    # Each model has exactly one atom of the first input group (enumerate_models
    # checks it), so the tasks' models are disjoint and together are all the models;
    # each task rejects repeated combinations among its own.
    context = multiprocessing.get_context("forkserver")
    # The server needs only this module, not the caller's main script.
    context.set_forkserver_preload([__name__])
    with ProcessPoolExecutor(options.processes, mp_context=context) as pool:
        parts = list(
            pool.map(
                task.enumerate_first,
                itertools.repeat(options.enum_mode),
                range(len(inputs[0])),
            )
        )
    rows = np.concatenate([part[0] for part in parts])
    columns = np.concatenate([part[1] for part in parts])
    models = group_models(
        rows, columns, combination_count(grounded.description), len(outputs)
    )
    return CompiledProgram(models, neural_atoms)


class _Grounded(NamedTuple):
    """A grounded program ready to enumerate: its generator, its inputs and the
    literals of its output atoms, in output-column order.
    """

    generator: ModuleGenerator
    description: InputDescription
    output_literals: list[int]


class _Task(NamedTuple):
    """What grounding a compiled program needs, picklable so that worker processes can
    ground it again: the plain ASP program, the input atom groups, the output atoms
    and Clingo's thread settings.
    """

    program: str
    inputs: list[list[str]]
    outputs: list[str]
    threads: int
    parallel_mode: str

    def ground(self) -> _Grounded:
        """Ground the program, showing its input and output atoms, and describe its
        inputs and outputs. Raises ValueError if an output atom is not grounded.
        """
        lib = Library()
        # Show the input and output predicates, so that each stable model is read
        # from those atoms alone (see ModuleGenerator). It must come before grounding;
        # the program's own #show directives, if any, still apply and are ignored.
        signatures = {
            symbol_signature(parse_symbol_term(lib, atom))
            for atom in [*(a for group in self.inputs for a in group), *self.outputs]
        }
        show = "\n".join(f"#show {name}/{arity}." for name, arity in sorted(signatures))
        control = Control(
            lib, ["0", f"--parallel-mode={self.threads},{self.parallel_mode}"]
        )
        control.parse_string(self.program + "\n" + show)
        control.ground()
        description = InputDescription.parse_inputs(lib, control, self.inputs)

        output_literals: list[int] = []
        for output in self.outputs:
            try:
                output_literals.append(symbol_id_from_string(lib, control, output)[0])
            except KeyError:
                raise ValueError(
                    f"Expected output atom {output} cannot be derived from the program."
                ) from None
        generator = ModuleGenerator(control, inputs_and_outputs_shown=True)
        return _Grounded(generator, description, output_literals)

    def enumerate_first(
        self, enum_mode: str, first: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ground the program and enumerate the models in which the first-th atom of
        the first input group is true; see ModuleGenerator.enumerate_models.
        """
        grounded = self.ground()
        # The first group's atoms come first in input_atoms, in the group's order.
        literal = grounded.description.input_atoms[first]
        return grounded.generator.enumerate_models(
            grounded.description, grounded.output_literals, enum_mode, [literal]
        )
