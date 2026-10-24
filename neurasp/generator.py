from array import array
import itertools
from math import prod
import operator
from typing import Generator, NamedTuple, Sequence

import numpy as np
import torch
from clingo import symbol
from clingo.core import Library
from clingo.control import Control
from clingo.ast import (
    parse_term,
    TermBinaryOperation,
    BinaryOperator,
    TermSymbolic,
    TermFunction,
)
from clingo.symbol import Symbol, SymbolType, parse_term as parse_symbol_term

from neurasp.module import NeurASPModule


def parse_signature(lib: Library, signature: str) -> tuple[str, int]:
    """Parse an ASP predicate signature such as ``"addition/3"`` into (name, arity).

    Raises ValueError if signature is not ``name/arity`` with a plain name and a number.
    """
    term = parse_term(lib, signature)
    if (
        not isinstance(term, TermBinaryOperation)
        or term.operator_type != BinaryOperator.Division
        or not isinstance(term.right, TermSymbolic)
        or term.right.symbol.type != SymbolType.Number
    ):
        raise ValueError(f"Invalid signature: {signature}")
    if (
        isinstance(term.left, TermSymbolic)
        and term.left.symbol.type == SymbolType.Function
    ):
        if term.left.symbol.arguments:
            raise ValueError(f"Invalid signature: {signature}")
        predicate_name = term.left.symbol.name
    elif isinstance(term.left, TermFunction):
        if term.left.pool and (len(term.left.pool) != 1 or term.left.pool[0].arguments):
            raise ValueError(f"Invalid signature: {signature}")
        predicate_name = term.left.name
    else:
        raise ValueError(f"Invalid signature: {signature}")
    arity = term.right.symbol.number
    return predicate_name, arity


def symbol_signature(symbol: symbol.Symbol) -> tuple[str, int]:
    """Return the (name, arity) signature of a function symbol. Raises ValueError for
    any other symbol, such as a number or a string, which cannot be an atom.
    """
    if symbol.type != SymbolType.Function:
        raise ValueError(f"Expected an atom such as name(arguments), got: {symbol}")
    return symbol.name, len(symbol.arguments)


def symbol_id_from_string(
    lib: Library, clingo_control: Control, atom_string: str
) -> tuple[int, Symbol]:
    """Return the program literal of the ground atom written as atom_string, and the
    atom's symbol. The atom must exist in the grounded program.
    """
    atom = parse_symbol_term(lib, atom_string)
    signature = symbol_signature(atom)
    return clingo_control.base[signature][atom].literal, atom


def symbol_ids_from_string(
    lib: Library, clingo_control: Control, atom_string: list[str]
) -> dict[int, Symbol]:
    """Map the program literal of each ground atom written in atom_string to its symbol
    (see symbol_id_from_string).
    """
    return dict(
        symbol_id_from_string(lib, clingo_control, atom) for atom in atom_string
    )


class InputDescription(NamedTuple):
    """The categorical inputs of a grounded program, as Clingo literals.

    input_atoms lists every input literal in the module's input-column order; inputs
    holds one group of literals per categorical input, exactly one of which is true in
    each stable model; atoms_map maps each literal to its symbol.
    """

    input_atoms: list[int]
    inputs: list[list[int]]
    atoms_map: dict[int, Symbol]

    @staticmethod
    def parse_inputs(
        lib: Library, clingo_control: Control, inputs: list[list[str]]
    ) -> "InputDescription":
        """Describe the categorical inputs from their atoms written as strings.

        inputs holds one group of atom strings per categorical input, such as the values
        of one neural output slot. input_atoms lists their literals in that order, which
        is the column order of the module's input; inputs holds each group's literals,
        sorted; atoms_map maps every literal to its symbol. Raises ValueError if two
        atoms of a group share a literal: the group would silently lose a column.
        """
        # One literal -> symbol map per group, with the literals in the group's order.
        groups = [
            symbol_ids_from_string(lib, clingo_control, group) for group in inputs
        ]
        for strings, group in zip(inputs, groups):
            if len(group) != len(strings):
                raise ValueError(
                    f"Input atoms {strings} share Clingo literals; each input atom "
                    "needs its own literal to be one input column"
                )
        atoms_map = {atom: symbol for group in groups for atom, symbol in group.items()}
        input_atoms = [atom for group in groups for atom in group]
        group_atoms = [sorted(group) for group in groups]
        return InputDescription(input_atoms, group_atoms, atoms_map)


# Clingo enumeration modes that enumerate every stable model (see generate_models).
ENUM_MODES = ("auto", "bt", "record")


# The errors for a stable model that does not have exactly one output atom, or exactly
# one atom of each input group.
ONE_OUTPUT = "Each model must have exactly one selected output atom"
ONE_INPUT_PER_GROUP = "Each model must select exactly one atom from each input group"


class Model(NamedTuple):
    """A stable model as its true output literals and, per input group, its true input
    literals.
    """

    output: frozenset[int]
    inputs: tuple[frozenset[int], ...]


class ModuleGenerator:
    """Enumerate the stable models of a grounded program and turn them into the models
    of a NeurASPModule.
    """

    def __init__(
        self, clingo_control: Control, inputs_and_outputs_shown: bool = False
    ) -> None:
        """Generate models from clingo_control, which must already be grounded.

        inputs_and_outputs_shown says that the program shows (``#show``) every input
        and output atom, as compile_program arranges. Each stable model is then read
        from its shown atoms, only a few, which is much faster than reading all its
        true atoms; without it, all true atoms are read, which is correct for any
        program.
        """
        self.control = clingo_control
        self.inputs_and_outputs_shown = inputs_and_outputs_shown

    def models(
        self,
        input_description: InputDescription,
        output_atoms: list[int],
        assumptions: Sequence[int] = (),
    ) -> Generator[Model, None, None]:
        """Solve and yield every stable model that satisfies assumptions (program
        literals assumed true) as its true output literals and, for each input group,
        its true input literals. Uses the control's current solve configuration (see
        enumerate_models for the enumeration settings).

        Each model's true atoms are read in one call and looked up by symbol, rather
        than asking the model about every input and output literal.
        """
        # Symbol -> (input group, literal), and output symbol -> literal.
        input_of = {
            input_description.atoms_map[literal]: (group, literal)
            for group, literals in enumerate(input_description.inputs)
            for literal in literals
        }
        wanted = set(output_atoms)
        output_of = {
            atom.symbol: atom.literal
            for atom_base in self.control.base.values()
            for atom in atom_base.values()
            if atom.literal in wanted
        }
        groups = len(input_description.inputs)
        shown = self.inputs_and_outputs_shown
        with self.control.start_solve(list(assumptions), yield_=True) as handle:
            for model in handle:
                outputs: set[int] = set()
                inputs: list[set[int]] = [set() for _ in range(groups)]
                for symbol in model.symbols(shown=shown, atoms=not shown):
                    hit = input_of.get(symbol)
                    if hit is not None:
                        inputs[hit[0]].add(hit[1])
                        continue
                    literal = output_of.get(symbol)
                    if literal is not None:
                        outputs.add(literal)
                yield Model(frozenset(outputs), tuple(frozenset(g) for g in inputs))

    @staticmethod
    def _validate_description(
        input_description: InputDescription,
        output_atoms: list[int],
    ) -> None:
        """Check input/output structure; definiteness is checked during compilation."""
        # Groups must be nonempty and disjoint: their Cartesian product describes all
        # allowed assignments, with one chosen atom from each group. The flat input
        # list must contain those same atoms exactly once; its order defines the
        # probability vector supplied to the returned PyTorch module.

        inputs = input_description.inputs
        if not inputs or any(not group for group in inputs):
            raise ValueError("Expected at least one nonempty input group")
        grouped_atoms = [atom for group in inputs for atom in group]
        if len(grouped_atoms) != len(set(grouped_atoms)):
            raise ValueError("Input groups must be disjoint and contain no duplicates")
        input_atoms = input_description.input_atoms
        if len(input_atoms) != len(set(input_atoms)) or set(input_atoms) != set(
            grouped_atoms
        ):
            raise ValueError(
                "input_atoms must contain each grouped input atom exactly once"
            )
        if not output_atoms or len(output_atoms) != len(set(output_atoms)):
            raise ValueError("Expected a nonempty list of distinct output atoms")

    def generate_module(
        self,
        input_description: InputDescription,
        output_atoms: list[int],
    ) -> NeurASPModule:
        """Compile the whole grounded program into a probability module.

        The returned module accepts batches of probabilities with columns in
        input_description.input_atoms order (one probability per domain atom). Its
        output columns follow output_atoms order. See generate_models() for the
        requirements on the program; the module itself never calls Clingo.
        """
        return NeurASPModule(self.generate_models(input_description, output_atoms))

    def generate_models(
        self,
        input_description: InputDescription,
        output_atoms: list[int],
        enum_mode: str = "auto",
    ) -> dict[int, torch.Tensor]:
        """Enumerate the stable models in the form NeurASPModule takes.

        Each input group is a categorical choice: exactly one of its atoms is true. This
        initial implementation requires exactly one stable model per input combination
        and exactly one selected output atom in each model. Unsupported inputs/programs
        raise ValueError; weights are not normalized.

        Returns output column (position in output_atoms) -> an int32 matrix, (models
        with that output, input groups), whose rows are the models in enumeration order
        and hold their selected atoms' positions in input_description.input_atoms.
        Enumeration covers all ordinary stable models, ignoring optimization and
        projection settings; see enumerate_models for enum_mode.
        """
        rows, columns = self.enumerate_models(
            input_description, output_atoms, enum_mode
        )
        return group_models(
            rows, columns, combination_count(input_description), len(output_atoms)
        )

    def enumerate_models(
        self,
        input_description: InputDescription,
        output_atoms: list[int],
        enum_mode: str = "auto",
        assumptions: Sequence[int] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enumerate the stable models that satisfy assumptions (program literals
        assumed true), as flat arrays in enumeration order: rows, an int32 (models,
        input groups) matrix of the selected atoms' positions in
        input_description.input_atoms, and columns, each model's output column
        (position in output_atoms).

        Raises ValueError for an invalid description, a model without exactly one
        output atom or one atom of each input group, or an input combination with
        several models. It does not check that every combination has a model:
        group_models does, once all the models are known.

        enum_mode is Clingo's enumeration mode, one of ENUM_MODES: "auto"
        (backtracking, Clingo's choice for this problem), "bt" (backtracking) or
        "record" (a nogood per model found, faster but using memory per model); all
        enumerate the same models. The control's other settings, such as its threads
        (--parallel-mode), apply as they are.
        """
        if enum_mode not in ENUM_MODES:
            raise ValueError(f"enum_mode must be one of {', '.join(ENUM_MODES)}")
        # 1. Check the categorical input description before asking Clingo to solve.
        self._validate_description(input_description, output_atoms)
        inputs = input_description.inputs

        # 2. One table, built once, so that each model needs a single dictionary
        # lookup per true atom: every input atom's symbol -> its group, its position
        # in input_description.input_atoms (its probability column) and its part of
        # the combination's code; every output atom's symbol -> its output column, in
        # output_atoms order (NeurASPModule's keys are columns 0, 1, ..., n - 1, not
        # Clingo literals).
        #
        # Every input combination has a code, its index in the grid of combinations
        # (a mixed-radix number with one digit per group), so one byte per combination
        # records which were seen: 7.3 MB for 52**4 combinations, instead of a set of
        # tuples. A model's code is the sum of its selected atoms' codes.
        position_of = {
            atom: index for index, atom in enumerate(input_description.input_atoms)
        }
        radix = itertools.accumulate(
            (len(g) for g in inputs[:-1]), operator.mul, initial=1
        )
        symbols = input_description.atoms_map
        input_of = {
            symbols[atom]: (group, position_of[atom], digit * group_radix)
            for (group, atoms), group_radix in zip(enumerate(inputs), radix)
            for digit, atom in enumerate(atoms)
        }
        column_of_literal = {atom: column for column, atom in enumerate(output_atoms)}
        column_of = {
            atom.symbol: column_of_literal[atom.literal]
            for atom_base in self.control.base.values()
            for atom in atom_base.values()
            if atom.literal in column_of_literal
        }
        groups = len(inputs)
        every_group = (1 << groups) - 1
        seen = bytearray(combination_count(input_description))
        # The models as flat arrays of C ints (int32): input positions, row by row, and
        # columns.
        positions = array("i")
        columns = array("i")

        # 3. Enumerate every ordinary stable model of the entire parsed program.
        solve = self.control.config.solve
        enumeration = {
            "models": "0",
            "enum_mode": enum_mode,
            "project": "no",
            "opt_mode": "ignore",
        }
        previous = {name: getattr(solve, name).value for name in enumeration}
        shown = self.inputs_and_outputs_shown
        try:
            for name, value in enumeration.items():
                setattr(solve, name, value)
            with self.control.start_solve(list(assumptions), yield_=True) as handle:
                for model in handle:
                    # Each model is read in one call and checked as it is read: one
                    # atom per input group (a bit per group already seen), exactly
                    # one output atom, and a combination not seen before.
                    row = [0] * groups
                    found = 0
                    code = 0
                    column = -1
                    for symbol in model.symbols(shown=shown, atoms=not shown):
                        hit = input_of.get(symbol)
                        if hit is not None:
                            group, position, part = hit
                            if found >> group & 1:
                                raise ValueError(ONE_INPUT_PER_GROUP)
                            found |= 1 << group
                            row[group] = position
                            code += part
                            continue
                        output = column_of.get(symbol)
                        if output is not None:
                            if column >= 0:
                                raise ValueError(ONE_OUTPUT)
                            column = output
                    if column < 0:
                        raise ValueError(ONE_OUTPUT)
                    if found != every_group:
                        raise ValueError(ONE_INPUT_PER_GROUP)
                    if seen[code]:
                        raise ValueError(
                            "An input combination has multiple stable models"
                        )
                    seen[code] = 1
                    positions.extend(row)
                    columns.append(column)
        finally:
            for name, value in previous.items():
                setattr(solve, name, value)

        column_array = np.frombuffer(columns, dtype=np.intc).astype(np.int32)
        rows = np.frombuffer(positions, dtype=np.intc).astype(np.int32)
        return rows.reshape(len(column_array), len(inputs)), column_array


def combination_count(input_description: InputDescription) -> int:
    """The number of input combinations: the product of the input groups' sizes."""
    return prod(len(group) for group in input_description.inputs)


def group_models(
    rows: np.ndarray, columns: np.ndarray, expected: int, output_count: int
) -> dict[int, torch.Tensor]:
    """Group enumerated models (see ModuleGenerator.enumerate_models) by output column,
    in the form NeurASPModule takes; see ModuleGenerator.generate_models.

    Raises ValueError unless there are exactly expected models, one per input
    combination: repeated combinations were already rejected, so this also checks that
    each one has a model.
    """
    if len(columns) != expected:
        raise ValueError(
            f"Expected {expected} input combinations, found {len(columns)}; "
            "the program must have exactly one stable model per combination"
        )
    # The stable sort keeps enumeration order within each output. The matrices are
    # views of one sorted tensor.
    order = np.argsort(columns, kind="stable")
    counts = np.bincount(columns, minlength=output_count).tolist()
    return dict(enumerate(torch.from_numpy(rows[order]).split(counts)))
