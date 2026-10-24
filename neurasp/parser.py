"""Parser for NeurASP neural atoms and their ASP/MVPP translation.

This module extracts the parsing behavior that was embedded in
``NeurASP.parse`` and ``NeurASP.nnAtom2MVPPrules``. It deliberately avoids
importing the original ``neurasp.py`` because that module pulls in PyTorch and
training code just to parse a program.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from clingo import ast
from clingo.control import Control
from clingo.core import Library
from clingo.symbol import Symbol


class ParseError(ValueError):
    """Raised when a NeurASP or MVPP fragment cannot be parsed."""


class MissingClingoError(RuntimeError):
    """Raised when the required Clingo 6 Python API is absent."""


@dataclass(frozen=True)
class ProbabilisticRule:
    """A non-neural MVPP probabilistic choice rule."""

    text: str
    probabilities: Tuple[float, ...]
    learnable: Tuple[bool, ...]
    atoms: Tuple[str, ...]
    choice_rule: str

    def __str__(self) -> str:
        """Return the rule's source text."""
        return self.text

    def to_dict(self) -> Dict[str, object]:
        """Return the rule as a JSON-serializable dictionary."""
        return {
            "text": self.text,
            "probabilities": list(self.probabilities),
            "learnable": list(self.learnable),
            "atoms": list(self.atoms),
            "choice_rule": self.choice_rule,
        }


@dataclass(frozen=True)
class MVPPParseResult:
    """The small part of MVPP.parse needed before grounding neural atoms."""

    probabilistic_rules: Tuple[ProbabilisticRule, ...]
    asp_program: str
    pi_prime: str

    @property
    def parameters(self) -> Tuple[Tuple[float, ...], ...]:
        """The probabilities of each probabilistic rule, in rule order."""
        return tuple(rule.probabilities for rule in self.probabilistic_rules)

    @property
    def learnable(self) -> Tuple[Tuple[bool, ...], ...]:
        """Whether each probability is learnable (written with ``@``), per rule."""
        return tuple(rule.learnable for rule in self.probabilistic_rules)


@dataclass(frozen=True)
class NeuralAtom:
    """A grounded nn(m(e,t), domain) declaration with ASP output slots 1..e."""

    raw: str
    network: str
    output_count: int
    term: str
    domain: Tuple[str, ...]

    @property
    def term_arity(self) -> int:
        """Number of top-level comma-separated arguments in term (0 if it is empty)."""
        return len(split_top_level(self.term)) if self.term else 0

    @property
    def output_width(self) -> int:
        """Width used by the original implementation for one output slot.

        Binary domains are represented by one neural probability and its complement, so
        the implementation width is 1 rather than 2.
        """

        if len(self.domain) == 2:
            return 1
        return len(self.domain)

    def __str__(self) -> str:
        """Return the grounded nn/2 atom as Clingo printed it."""
        return self.raw

    def to_dict(self) -> Dict[str, object]:
        """Return the atom and its derived properties as a JSON-serializable dict."""
        return {
            "raw": self.raw,
            "network": self.network,
            "output_count": self.output_count,
            "term": self.term,
            "term_arity": self.term_arity,
            "domain": list(self.domain),
            "output_width": self.output_width,
        }


@dataclass(frozen=True)
class ParsedNeurASPProgram:
    """Parsed NeurASP artifacts needed by downstream ASP tasks."""

    standard_asp_program: str
    neural_atoms: Tuple[NeuralAtom, ...]

    def to_dict(self) -> Dict[str, object]:
        """Return the program and its neural atoms as a JSON-serializable dictionary."""
        return {
            "standard_asp_program": self.standard_asp_program,
            "neural_atoms": [atom.to_dict() for atom in self.neural_atoms],
        }

    def to_json(self, **json_kwargs: Any) -> str:
        """Return to_dict() as JSON; json_kwargs go to json.dumps (indent defaults to
        2).
        """
        kwargs: Dict[str, Any] = {"indent": 2}
        kwargs.update(json_kwargs)
        return json.dumps(self.to_dict(), **kwargs)

    def __str__(self) -> str:
        """Return the standard ASP program."""
        return self.standard_asp_program


def parse_neurasp_program(
    program: Union[str, os.PathLike[str]],
    obs: str = "",
    *,
    clingo_options: Optional[Sequence[str]] = None,
    grounder: Optional[Callable[[str], Iterable[object]]] = None,
) -> ParsedNeurASPProgram:
    """Parse a NeurASP program into ASP and neural-atom artifacts.

    Args:
        program: Either a NeurASP program string or a path to a program file.
        obs: Optional observation program appended before parsing, matching the original
            NeurASP API.
        clingo_options: Options passed to ``clingo.Control`` when grounding.
        grounder: Test hook. When supplied, it receives the grounding program and
            returns grounded clingo symbols; otherwise clingo is used.

    Returns:
        A ``ParsedNeurASPProgram`` containing the standard ASP counterpart and
        grounded neural atoms.
    """

    source_program = _read_program(program) + obs
    constants = parse_constants(source_program)
    grounding_source = prepare_program_for_grounding(source_program)
    mvpp_parse = parse_mvpp_choices(grounding_source)

    symbols = (
        list(grounder(mvpp_parse.pi_prime))
        if grounder
        else _ground_symbols(mvpp_parse.pi_prime, clingo_options)
    )
    neural_atoms = tuple(
        parse_ground_neural_atom(symbol, constants)
        for symbol in symbols
        if getattr(symbol, "name", None) == "nn"
    )

    asp_program = remove_neural_declaration_lines(source_program)
    neural_choice_rules = tuple(
        rule for atom in neural_atoms for rule in neural_atom_to_choice_rules(atom)
    )
    standard_asp_program = build_standard_asp_program(
        asp_program,
        neural_choice_rules,
    )

    return ParsedNeurASPProgram(
        standard_asp_program=standard_asp_program,
        neural_atoms=neural_atoms,
    )


def parse_constants(program: str) -> Dict[str, str]:
    """Return all ``#const name = value.`` definitions in source order."""

    constants: Dict[str, str] = {}
    regex = re.compile(r"(?m)^\s*#const\s+([^=\s]+)\s*=\s*([^.]+)\.")
    for match in regex.finditer(program):
        constants[match.group(1).strip()] = match.group(2).strip()
    return constants


def prepare_program_for_grounding(program: str) -> str:
    """Prepare source for grounding neural declarations.

    Drop whole comment lines (``%``), so the line-based MVPP parsing does not read
    them as rules, and rewrite the neural domains with bracket_neural_domains. Weak
    constraints are kept: their ``[w@p]`` brackets are left alone, and enumeration
    ignores optimization.
    """

    kept_lines = [
        line for line in program.splitlines() if not line.lstrip().startswith("%")
    ]
    return bracket_neural_domains("\n".join(kept_lines))


_NEURAL_DECLARATION = re.compile(r"(?<![\w'])nn\(")


def bracket_neural_domains(program: str) -> str:
    """Replace square brackets with parentheses inside ``nn(...)`` declarations only.

    NeurASP writes neural domains as lists, such as ``nn(digit(1,X), [0,1])``, which
    is not plain ASP; as a tuple, ``(0,1)``, Clingo reads it as an ordinary term.
    Brackets elsewhere, such as the ``[w@p]`` of a weak constraint, and inside quoted
    strings are kept.
    """
    characters = list(program)
    for match in _NEURAL_DECLARATION.finditer(program):
        depth = 0
        quote: Optional[str] = None
        escaped = False
        for index in range(match.end() - 1, len(program)):
            char = program[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char == '"':
                quote = char
            elif char in "([":
                depth += 1
                if char == "[":
                    characters[index] = "("
            elif char in ")]":
                depth -= 1
                if char == "]":
                    characters[index] = ")"
                if depth == 0:
                    break
    return "".join(characters)


def parse_mvpp_choices(program: Union[str, os.PathLike[str]]) -> MVPPParseResult:
    """Split MVPP probabilistic choice rules from the plain ASP lines.

    Each probabilistic rule, such as ``0.3 a; @0.7 b.``, becomes the Clingo choice rule
    ``1{a; b}1.``; every line is stripped. pi_prime holds those choice rules followed by
    the ASP lines. program is source text or a path to a file.
    """

    source = _read_program(program)
    asp_lines: List[str] = []
    choice_lines: List[str] = []
    probabilistic_rules: List[ProbabilisticRule] = []

    for line in source.splitlines():
        stripped = line.strip()
        if is_probabilistic_choice_rule(stripped):
            rule = parse_probabilistic_rule(stripped)
            probabilistic_rules.append(rule)
            choice_lines.append(rule.choice_rule)
        else:
            asp_lines.append(stripped)

    asp_program = join_program_lines(asp_lines)
    pi_prime = join_program_lines(choice_lines + ([asp_program] if asp_program else []))
    return MVPPParseResult(
        probabilistic_rules=tuple(probabilistic_rules),
        asp_program=asp_program,
        pi_prime=pi_prime,
    )


def build_standard_asp_program(
    asp_program: str, neural_choice_rules: Sequence[str]
) -> str:
    """Return a clingo ASP counterpart with probabilistic syntax removed."""

    parsed_asp = parse_mvpp_choices(asp_program)
    choice_lines = list(neural_choice_rules)
    choice_lines.extend(rule.choice_rule for rule in parsed_asp.probabilistic_rules)
    if parsed_asp.asp_program:
        choice_lines.append(parsed_asp.asp_program)
    return join_program_lines(choice_lines)


def is_probabilistic_choice_rule(line: str) -> bool:
    """Whether line looks like an MVPP probabilistic choice rule: it starts with a
    probability (``@``-prefixed if learnable) and an atom, contains ``;`` and ends with
    ``.``.
    """
    if not line or ";" not in line:
        return False
    return bool(
        re.match(
            r"^@?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s+.+;.+\.\s*$",
            line,
        )
    )


def parse_probabilistic_rule(line: str) -> ProbabilisticRule:
    """Parse an MVPP probabilistic choice rule such as ``@0.2 a(1); @0.8 b(1).``.

    Returns its probabilities, learnable flags, normalized atoms and the Clingo choice
    rule ``1{a(1); b(1)}1.``. Raises ParseError on a malformed choice or probability.
    """
    if not line.endswith("."):
        raise ParseError(f"Probabilistic rule must end with '.': {line}")

    atoms: List[str] = []
    probabilities: List[float] = []
    learnable: List[bool] = []
    for choice in split_top_level(line[:-1], delimiter=";"):
        parts = choice.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ParseError(f"Invalid probabilistic choice: {choice}")
        prob_token, atom = parts
        is_learnable = prob_token.startswith("@")
        prob_text = prob_token[1:] if is_learnable else prob_token
        try:
            probability = float(prob_text)
        except ValueError as exc:
            raise ParseError(f"Invalid probability {prob_token!r} in {line}") from exc
        probabilities.append(probability)
        learnable.append(is_learnable)
        atoms.append(normalize_probabilistic_atom(atom))

    choice_rule = "1{" + "; ".join(atoms) + "}1."
    return ProbabilisticRule(
        text=line,
        probabilities=tuple(probabilities),
        learnable=tuple(learnable),
        atoms=tuple(atoms),
        choice_rule=choice_rule,
    )


def normalize_probabilistic_atom(atom: str) -> str:
    """Return a canonical atom string for an MVPP probabilistic choice."""

    try:
        statements = parse_asp_statements(atom.strip() + ".")
    except (MissingClingoError, ParseError):
        return atom.replace(" ", "")

    for statement in statements:
        if not _is_rule_statement(statement):
            continue
        if getattr(statement, "body", None):
            continue
        literal = _head_literal(statement)
        if literal is not None:
            return str(literal).replace(" ", "")
    return atom.replace(" ", "")


def parse_ground_neural_atom(
    atom: object, constants: Optional[Dict[str, str]] = None
) -> NeuralAtom:
    """Parse a grounded Clingo symbol such as ``nn(digit(1,i),(0,1))`` into a
    NeuralAtom.

    constants (from ``#const`` definitions) are substituted in the output count and the
    term. Raises ParseError if atom is not a well-formed nn/2 atom with at least two
    domain values.
    """

    constants = constants or {}
    raw = str(atom).strip()
    if getattr(atom, "name", None) != "nn":
        raise ParseError(f"Expected an nn/2 atom, got: {raw}")

    args = _symbol_arguments(atom)
    if len(args) != 2:
        raise ParseError(f"Expected nn/2 atom, got {len(args)} arguments: {raw}")

    network_term = args[0]
    network = getattr(network_term, "name", None)
    if not network:
        raise ParseError(f"Expected neural term m(e,t), got: {network_term}")

    network_args = _symbol_arguments(network_term)
    if len(network_args) < 2:
        raise ParseError(f"Expected neural term m(e,t), got: {network_term}")

    output_count = parse_output_count(str(network_args[0]), constants)
    term = ",".join(replace_constant(str(part), constants) for part in network_args[1:])
    domain = tuple(_symbol_sequence_to_strings(args[1]))
    if len(domain) < 2:
        raise ParseError(f"Neural atom domain must contain at least 2 values: {raw}")

    return NeuralAtom(
        raw=raw,
        network=network,
        output_count=output_count,
        term=term,
        domain=domain,
    )


def parse_output_count(text: str, constants: Dict[str, str]) -> int:
    """Return the output count of a neural term after substituting constants.

    Raises ParseError if it is not an integer, including the original NeurASP's
    ``yolo`` (object detection with a varying number of outputs), which is not
    supported.
    """
    output_count_text = replace_constant(text, constants)
    if output_count_text == "yolo":
        raise ParseError("yolo neural atoms (object detection) are not supported")
    try:
        return int(output_count_text)
    except ValueError as exc:
        raise ParseError(
            f"Neural output count must be an integer: {output_count_text}"
        ) from exc


def _symbol_arguments(symbol: object) -> Tuple[object, ...]:
    """Return a symbol's arguments, or an empty tuple if it has none."""
    args = getattr(symbol, "arguments", None)
    if args is None:
        return tuple()
    return tuple(args)


def _symbol_sequence_to_strings(symbol: object) -> Tuple[str, ...]:
    """Return the values of a domain symbol as strings: the arguments of a tuple or
    function, or else the top-level comma-separated parts of its text without outer
    brackets.
    """
    args = _symbol_arguments(symbol)
    if args:
        return tuple(str(arg) for arg in args)

    text = str(symbol).strip()
    text = strip_outer(text, ("(", ")"))
    text = strip_outer(text, ("[", "]"))
    return tuple(split_top_level(text))


def neural_atom_to_choice_rules(atom: NeuralAtom) -> Tuple[str, ...]:
    """Translate one grounded neural atom directly to standard ASP choices.

    There is one rule per output slot 1..e, ``1{m(i, t, v1); ...; m(i, t, vn)}1.``,
    choosing exactly one domain value.
    """

    rules: List[str] = []
    for output_index in range(1, atom.output_count + 1):
        atoms = [
            format_network_atom(atom.network, output_index, atom.term, value).replace(
                " ", ""
            )
            for value in atom.domain
        ]
        rules.append("1{" + "; ".join(atoms) + "}1.")
    return tuple(rules)


def remove_neural_declaration_lines(program: str) -> str:
    """Return the source program with neural declaration rules removed.

    Clingo's AST parser handles the ASP part structurally. MVPP probabilistic rules are
    separated first because their ``@`` syntax is not plain ASP. NeurASP neural domains
    are rewritten with bracket_neural_domains before AST parsing because declarations
    such as ``nn(digit(1,X), [0,1])`` are not plain ASP either.
    """

    parsed_mvpp = parse_mvpp_choices(program)
    ast_program = bracket_neural_domains(parsed_mvpp.asp_program)
    kept_statements = [
        ast_to_source(statement)
        for statement in parse_asp_statements(ast_program)
        if not is_neural_declaration_statement(statement)
    ]
    return join_program_lines(
        [str(rule) for rule in parsed_mvpp.probabilistic_rules] + kept_statements
    )


def parse_asp_statements(program: str) -> Tuple[object, ...]:
    """Parse ASP source into Clingo AST statements."""

    if not program.strip():
        return tuple()

    library = Library()

    statements: List[object] = []
    try:
        ast.parse_string(library, program, statements.append)
    except RuntimeError as exc:
        raise ParseError(f"Could not parse ASP program with clingo AST: {exc}") from exc
    return tuple(statements)


def is_neural_declaration_statement(statement: object) -> bool:
    """Return True for rules whose head is structurally an ``nn(...)`` atom."""

    if not _is_rule_statement(statement):
        return False

    literal = _head_literal(statement)
    atom = _symbolic_literal_atom(literal)
    return _is_ast_function_named(atom, "nn")


def ast_to_source(statement: object) -> str:
    """Render a Clingo AST statement to source, skipping implicit directives."""

    if _ast_node_name(statement) in {"Program", "StatementProgram"}:
        return ""
    return str(statement).strip()


def _is_rule_statement(statement: object) -> bool:
    """Whether an AST statement is a rule (Clingo 5 or 6 node name)."""
    return _ast_node_name(statement) in {"Rule", "StatementRule"}


def _head_literal(statement: object) -> Optional[object]:
    """Return the literal of a rule's simple-literal head, or None for other heads."""
    head = getattr(statement, "head", None)
    if head is None:
        return None
    if _ast_node_name(head) in {"Literal", "HeadSimpleLiteral"}:
        return getattr(head, "literal", head)
    return None


def _symbolic_literal_atom(literal: object) -> Optional[object]:
    """Return the atom of a symbolic literal, or None for other literals."""
    if literal is None:
        return None
    if _ast_node_name(literal) in {"LiteralSymbolic", "SymbolicAtom"}:
        return getattr(literal, "atom", getattr(literal, "symbol", None))
    if _ast_node_name(literal) == "Literal":
        atom = getattr(literal, "atom", None)
        if _ast_node_name(atom) == "SymbolicAtom":
            return getattr(atom, "symbol", None)
    return None


def _is_ast_function_named(node: object, name: str) -> bool:
    """Whether an AST node is a function term called name."""
    return (
        node is not None
        and _ast_node_name(node) in {"Function", "TermFunction"}
        and getattr(node, "name", None) == name
    )


def _ast_node_name(node: object) -> str:
    """Return the type name of a Clingo AST node: its class name for Clingo 6 nodes, or
    the ast_type name of a generic Clingo 5 AST; ``"None"`` for None.
    """
    if node is None:
        return "None"
    class_name = type(node).__name__
    if class_name != "AST":
        return class_name
    ast_type = getattr(node, "ast_type", None)
    if ast_type is None:
        ast_type = getattr(node, "type", None)
    name = getattr(ast_type, "name", None)
    if name is not None:
        return name
    return str(ast_type).rsplit(".", 1)[-1]


def split_top_level(text: str, delimiter: str = ",") -> List[str]:
    """Split on a delimiter that is not inside quotes or brackets."""

    parts: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    escaped = False

    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ParseError(f"Unbalanced closing bracket in: {text}")
        elif char == delimiter and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1

    if quote:
        raise ParseError(f"Unclosed quote in: {text}")
    if depth != 0:
        raise ParseError(f"Unbalanced brackets in: {text}")

    tail = text[start:].strip()
    if tail or text:
        parts.append(tail)
    return parts


def strip_outer(text: str, pair: Tuple[str, str]) -> str:
    """Strip text and remove one enclosing pair of the given brackets, if present."""
    text = text.strip()
    if text.startswith(pair[0]) and text.endswith(pair[1]):
        return text[1:-1].strip()
    return text


def replace_constant(text: str, constants: Dict[str, str]) -> str:
    """Return the value of the constant named by text, or text itself (stripped)."""
    stripped = text.strip()
    return constants.get(stripped, stripped)


def replace_constants_in_csv(text: str, constants: Dict[str, str]) -> str:
    """Replace constants in each top-level comma-separated part of text."""
    return ",".join(replace_constant(part, constants) for part in split_top_level(text))


def format_network_atom(network: str, output_index: int, term: str, value: str) -> str:
    """Format an ASP neural prediction using a one-based output slot."""
    return f"{network}({output_index}, {term}, {value})"


def join_program_lines(lines: Iterable[str]) -> str:
    """Join lines into a program, without trailing whitespace or leading and trailing
    blank lines.
    """
    cleaned = [line.rstrip() for line in lines]
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned)


def _read_program(program: Union[str, os.PathLike[str]]) -> str:
    """Return program text: the contents of a path (a PathLike, or a string naming an
    existing file), or else the string itself.
    """
    if isinstance(program, os.PathLike):
        return Path(program).read_text()
    if isinstance(program, str) and os.path.isfile(program):
        return Path(program).read_text()
    if isinstance(program, str):
        return program
    raise TypeError(f"Expected program text or path, got {type(program)!r}")


def _ground_symbols(
    program: str, clingo_options: Optional[Sequence[str]] = None
) -> List[Symbol]:
    """Ground program with Clingo (and clingo_options) and return its symbolic atoms."""
    library = Library()
    options = list(clingo_options or [])
    control = Control(library, options)
    control.parse_string(program)
    control.ground()
    return symbols_from_control_base(control)


def symbols_from_control_base(control: object) -> List[Symbol]:
    """Return symbolic atoms from a Clingo 6 control object's atom base."""

    base = getattr(control, "base", None)
    if base is None:
        raise MissingClingoError(
            "Clingo 6 control object does not expose `base`; cannot inspect "
            "ground symbolic atoms."
        )

    symbols: List[Symbol] = []
    for atom_base in base.values():
        symbols.extend(atom.symbol for atom in atom_base.values())
    return symbols
