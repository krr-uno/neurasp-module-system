import textwrap
import unittest

import torch

from clingo.control import Control
from clingo.core import Library
from clingo.symbol import Symbol, parse_term as parse_symbol_term

from neurasp.parser import parse_neurasp_program
from neurasp.parser import format_network_atom

from neurasp.generator import InputDescription, ModuleGenerator
from neurasp.module import NeurASPModule

# class TestAtomBase(unittest.TestCase):
#     def test_atom_base(self):

#         lib = Library()
#         control = Control(lib)
#         control.parse_string("a. a(1..2). a(3,4).")
#         control.ground()
#         for atom in control.base[("a", 1)].values():
#             print(atom.literal, atom.symbol)


def get_literal_map(control: Control) -> dict[int, Symbol]:
    """Return a dictionary mapping atom literals to their corresponding symbols."""
    return {
        atom.literal: atom.symbol
        for atom_base in control.base.values()
        for atom in atom_base.values()
    }


class TestInputDescription(unittest.TestCase):
    """InputDescription.parse_inputs."""

    def test_input_description(self) -> None:
        """parse_inputs keeps the groups and order of the atom strings it is given."""
        from clingo.control import Control
        from clingo.core import Library

        lib = Library()
        control = Control(lib)
        control.parse_string("{digit(1,X,0..9)} = 1 :- X=1..2.")
        control.ground()
        str_atoms = [
            [f"digit(1,1,{i})" for i in range(10)],
            [f"digit(1,2,{i})" for i in range(10)],
        ]
        input_description = InputDescription.parse_inputs(lib, control, str_atoms)
        self.assertEqual(len(input_description.inputs), 2)
        result: list[list[str]] = []
        for inputs in input_description.inputs:
            self.assertEqual(len(inputs), 10)
            atoms: list[str] = []
            for atom in inputs:
                self.assertIsInstance(atom, int)
                self.assertIsInstance(input_description.atoms_map[atom], Symbol)
                atoms.append(str(input_description.atoms_map[atom]))
            result.append(atoms)
        self.assertEqual(result, str_atoms)

    def test_input_atoms_sharing_a_literal_are_rejected(self) -> None:
        """Two atoms of a group with one literal would merge into one input column."""
        lib = Library()

        class Atom:
            """A ground atom of the stand-in base: just its literal."""

            def __init__(self, literal: int) -> None:
                """An atom with the given literal."""
                self.literal = literal

        class SharedLiterals:
            """A stand-in for a grounded Control whose atoms d(1,a,0) and d(1,a,1)
            share literal 3.
            """

            base = {
                ("d", 3): {
                    parse_symbol_term(lib, "d(1,a,0)"): Atom(3),
                    parse_symbol_term(lib, "d(1,a,1)"): Atom(3),
                }
            }

        with self.assertRaisesRegex(ValueError, "share Clingo literals"):
            InputDescription.parse_inputs(
                lib,
                SharedLiterals(),  # type: ignore[arg-type]
                [["d(1,a,0)", "d(1,a,1)"]],
            )


class TestModuleGenerator(unittest.TestCase):
    """Model enumeration and module generation from parsed programs."""

    NEURASP_ADDITION = textwrap.dedent("""
    img(1..{images}).
    nn(digit(1,X), [{domain}]) :- img(X).
    sum(2,S+V) :- digit(1,1,S), digit(1,2,V).
    sum(D,S+V) :- sum(D-1,S), digit(1,D,V).
    """)

    def parse_module(self, program: str) -> tuple[Control, InputDescription]:
        """Parse and ground program; return the control and one input group per neural
        output slot.
        """
        parsed = parse_neurasp_program(program)
        self.assertTrue(parsed.neural_atoms)
        lib = Library()
        control = Control(lib, ["0"])
        control.parse_string(parsed.standard_asp_program)
        control.ground()

        # Each neural output slot is one categorical input to the generator.
        inputs = []
        for atom in parsed.neural_atoms:
            for output_index in range(1, atom.output_count + 1):
                inputs.append(
                    [
                        format_network_atom(
                            atom.network, output_index, atom.term, value
                        )
                        for value in atom.domain
                    ]
                )
        return control, InputDescription.parse_inputs(lib, control, inputs)

    def parse_addition(
        self, possible_digits: int, number_of_images: int
    ) -> tuple[Control, InputDescription]:
        """Parse and ground the addition of number_of_images images with digits
        0..possible_digits - 1.
        """
        program = self.NEURASP_ADDITION.format(
            images=number_of_images,
            domain=",".join(str(value) for value in range(possible_digits)),
        )
        return self.parse_module(program)

    def test_mnist_2_3_models(self) -> None:
        """Adding three binary digits has 8 stable models, each with one input per
        image.
        """
        control, input_description = self.parse_addition(2, 3)
        symbol_map = get_literal_map(control)
        output_atoms = [
            atom.literal
            for atom in control.base[("sum", 2)].values()
            if atom.symbol.arguments[0].number == 3
        ]
        models = list(ModuleGenerator(control).models(input_description, output_atoms))
        self.assertEqual(len(input_description.inputs), 3)
        self.assertEqual(len(models), 8)
        for model in models:
            self.assertEqual(len(model.output), 1)
            self.assertEqual(len(model.inputs), 3)
            self.assertTrue(all(len(group) == 1 for group in model.inputs))
        symbolic_models = [
            (
                frozenset(str(symbol_map[literal]) for literal in output),
                tuple(
                    frozenset(str(symbol_map[literal]) for literal in input)
                    for input in inputs
                ),
            )
            for output, inputs in models
        ]
        # The three images are independent binary choices, so the models are
        # exactly the eight triples and their corresponding sums.
        self.assertCountEqual(
            symbolic_models,
            [
                (
                    frozenset({"sum(3,0)"}),
                    (
                        frozenset({"digit(1,1,0)"}),
                        frozenset({"digit(1,2,0)"}),
                        frozenset({"digit(1,3,0)"}),
                    ),
                ),
                (
                    frozenset({"sum(3,1)"}),
                    (
                        frozenset({"digit(1,1,0)"}),
                        frozenset({"digit(1,2,0)"}),
                        frozenset({"digit(1,3,1)"}),
                    ),
                ),
                (
                    frozenset({"sum(3,1)"}),
                    (
                        frozenset({"digit(1,1,0)"}),
                        frozenset({"digit(1,2,1)"}),
                        frozenset({"digit(1,3,0)"}),
                    ),
                ),
                (
                    frozenset({"sum(3,1)"}),
                    (
                        frozenset({"digit(1,1,1)"}),
                        frozenset({"digit(1,2,0)"}),
                        frozenset({"digit(1,3,0)"}),
                    ),
                ),
                (
                    frozenset({"sum(3,2)"}),
                    (
                        frozenset({"digit(1,1,0)"}),
                        frozenset({"digit(1,2,1)"}),
                        frozenset({"digit(1,3,1)"}),
                    ),
                ),
                (
                    frozenset({"sum(3,2)"}),
                    (
                        frozenset({"digit(1,1,1)"}),
                        frozenset({"digit(1,2,0)"}),
                        frozenset({"digit(1,3,1)"}),
                    ),
                ),
                (
                    frozenset({"sum(3,2)"}),
                    (
                        frozenset({"digit(1,1,1)"}),
                        frozenset({"digit(1,2,1)"}),
                        frozenset({"digit(1,3,0)"}),
                    ),
                ),
                (
                    frozenset({"sum(3,3)"}),
                    (
                        frozenset({"digit(1,1,1)"}),
                        frozenset({"digit(1,2,1)"}),
                        frozenset({"digit(1,3,1)"}),
                    ),
                ),
            ],
        )

    def test_mnist_2_2_module_generator(self) -> None:
        """Adding two binary digits gives exactly the four digit pairs as models."""
        control, input_description = self.parse_addition(2, 2)
        symbol_map = get_literal_map(control)
        output_atoms = [atom.literal for atom in control.base[("sum", 2)].values()]
        module_generator = ModuleGenerator(control)
        models = list(module_generator.models(input_description, output_atoms))
        self.assertEqual(len(models), 4)
        symbolic_models = [
            (
                frozenset(str(symbol_map[literal]) for literal in output),
                tuple(
                    frozenset(str(symbol_map[literal]) for literal in input)
                    for input in inputs
                ),
            )
            for output, inputs in models
        ]
        # The two images are independent choices, so the models are exactly the
        # four digit pairs. Every model carries one input atom per image and a
        # single output atom; sum(2,1) shows up twice, for 1+0 and for 0+1.
        self.assertCountEqual(
            symbolic_models,
            [
                (
                    frozenset({"sum(2,0)"}),
                    (frozenset({"digit(1,1,0)"}), frozenset({"digit(1,2,0)"})),
                ),
                (
                    frozenset({"sum(2,1)"}),
                    (frozenset({"digit(1,1,0)"}), frozenset({"digit(1,2,1)"})),
                ),
                (
                    frozenset({"sum(2,1)"}),
                    (frozenset({"digit(1,1,1)"}), frozenset({"digit(1,2,0)"})),
                ),
                (
                    frozenset({"sum(2,2)"}),
                    (frozenset({"digit(1,1,1)"}), frozenset({"digit(1,2,1)"})),
                ),
            ],
        )

    def test_multiple_neural_output_slots(self) -> None:
        """An atom with two output slots gives two input groups and 3x3 models."""
        control, input_description = self.parse_module("""
            nn(label(2,item(1)), [blue,red,green]).
            pair(A,B) :- label(1,item(1),A), label(2,item(1),B).
        """)
        self.assertEqual(
            [
                {str(input_description.atoms_map[literal]) for literal in group}
                for group in input_description.inputs
            ],
            [
                {
                    "label(1,item(1),blue)",
                    "label(1,item(1),red)",
                    "label(1,item(1),green)",
                },
                {
                    "label(2,item(1),blue)",
                    "label(2,item(1),red)",
                    "label(2,item(1),green)",
                },
            ],
        )
        output_atoms = [atom.literal for atom in control.base[("pair", 2)].values()]
        models = list(ModuleGenerator(control).models(input_description, output_atoms))
        self.assertEqual(len(models), 9)
        for model in models:
            self.assertEqual(len(model.output), 1)
            self.assertEqual(len(model.inputs), 2)
            self.assertTrue(all(len(group) == 1 for group in model.inputs))

    # Tests for ModuleGenerator.generate_module()

    def output_literals(self, control: Control, *atoms: str) -> list[int]:
        """Return the literals of the named ground atoms, in the given order.

        Names make the requested output column order explicit; the literals themselves
        still come from grounding the parser's ASP translation.
        """
        literals = {
            str(atom.symbol): atom.literal
            for atom_base in control.base.values()
            for atom in atom_base.values()
        }
        return [literals[atom] for atom in atoms]

    def test_generate_module_probabilities_and_gradients(self) -> None:
        """The module computes sum probabilities and their gradients exactly."""
        control, inputs = self.parse_addition(2, 2)
        outputs = self.output_literals(control, "sum(2,0)", "sum(2,1)", "sum(2,2)")
        module = ModuleGenerator(control).generate_module(inputs, outputs)
        self.assertIsInstance(module, NeurASPModule)
        self.assertEqual(
            [str(inputs.atoms_map[literal]) for literal in inputs.input_atoms],
            ["digit(1,1,0)", "digit(1,1,1)", "digit(1,2,0)", "digit(1,2,1)"],
        )
        probabilities = torch.tensor(
            [[0.8, 0.2, 0.3, 0.7], [0.1, 0.9, 0.6, 0.4]], requires_grad=True
        )

        # With columns [a0, a1, b0, b1], the three sums have probabilities
        # [a0*b0, a0*b1 + a1*b0, a1*b1]. In particular, sum=1 must add
        # two different stable models instead of retaining only one of them.
        result = module(probabilities)
        torch.testing.assert_close(
            result, torch.tensor([[0.24, 0.62, 0.14], [0.06, 0.58, 0.36]])
        )
        torch.testing.assert_close(result.sum(dim=1), torch.ones(2))

        # The derivative of P(sum=1) is [b1, b0, a1, a0]. Checking it
        # verifies that compiling the logic preserves the training path.
        result[:, 1].sum().backward()
        torch.testing.assert_close(
            probabilities.grad,
            torch.tensor([[0.7, 0.3, 0.2, 0.8], [0.4, 0.6, 0.9, 0.1]]),
        )

    def test_generate_module_preserves_input_and_output_column_order(self) -> None:
        """Columns follow the given input_atoms and output order, not Clingo's."""
        control, inputs = self.parse_module("""
            nn(label(1,item), [blue,red,green]).
            out(C) :- label(1,item,C).
        """)

        blue, red, green = inputs.input_atoms
        inputs = inputs._replace(input_atoms=[green, blue, red])
        outputs = self.output_literals(control, "out(red)", "out(green)", "out(blue)")
        module = ModuleGenerator(control).generate_module(inputs, outputs)
        torch.testing.assert_close(
            module(torch.tensor([[0.5, 0.2, 0.3]])), torch.tensor([[0.3, 0.5, 0.2]])
        )

    def test_generate_module_multiple_slots_and_unequal_domains(self) -> None:
        """Slots of one atom and domains of different sizes combine correctly."""
        control, inputs = self.parse_module("""
            nn(digit(2,item), [0,1]).
            nn(offset(1,item), [0,1,2]).
            total(A+B+C) :- digit(1,item,A), digit(2,item,B), offset(1,item,C).
        """)
        self.assertEqual([len(group) for group in inputs.inputs], [2, 2, 3])
        outputs = self.output_literals(
            control, *(f"total({value})" for value in range(5))
        )
        module = ModuleGenerator(control).generate_module(inputs, outputs)

        # The first two slots produce [0.24, 0.62, 0.14]. Combining that
        # distribution with offset probabilities [0.1, 0.6, 0.3] gives the
        # five coefficients below, over 2*2*3=12 stable models.
        self.assertEqual(tuple(module.models.shape), (12, 3))
        torch.testing.assert_close(
            module(torch.tensor([[0.8, 0.2, 0.3, 0.7, 0.1, 0.6, 0.3]])),
            torch.tensor([[0.024, 0.206, 0.458, 0.270, 0.042]]),
        )

    def test_generate_module_enumerates_all_models_and_restores_limit(self) -> None:
        """All models are enumerated even if the control is set to find one, and the
        setting is restored.
        """
        control, inputs = self.parse_addition(2, 2)
        control.config.solve.models = "1"
        outputs = self.output_literals(control, "sum(2,0)", "sum(2,1)", "sum(2,2)")
        module = ModuleGenerator(control).generate_module(inputs, outputs)

        # A solver configured to find one answer still needs all four digit
        # assignments for probability compilation, then regains its setting.
        self.assertEqual(control.config.solve.models.value, "1")
        self.assertEqual(tuple(module.models.shape), (4, 2))
        torch.testing.assert_close(
            module(torch.tensor([[0.5, 0.5, 0.5, 0.5]])),
            torch.tensor([[0.25, 0.5, 0.25]]),
        )

    def test_generate_module_keeps_unreachable_output_column(self) -> None:
        """An output that is in no stable model keeps its zero-probability column."""
        control, inputs = self.parse_module("""
            nn(label(1,item), [blue,red]).
            out(C) :- label(1,item,C).
            {out(unreachable)}.
            :- out(unreachable).
        """)
        outputs = self.output_literals(
            control, "out(unreachable)", "out(blue)", "out(red)"
        )
        module = ModuleGenerator(control).generate_module(inputs, outputs)

        input_probabilities = torch.tensor([[0.25, 0.75]])
        # Output columns follow the requested order: [unreachable, blue, red].
        expected_output_probabilities = torch.tensor([[0.0, 0.25, 0.75]])

        # An output can exist in the grounded atom table without appearing
        # in any stable model. Its requested column must remain present.

        actual_output_probabilities = module(input_probabilities)
        torch.testing.assert_close(
            actual_output_probabilities, expected_output_probabilities
        )

    def test_generate_module_requires_one_selected_output_per_model(self) -> None:
        """Models with no output or two outputs are rejected."""
        for output_names in [
            ["sum(2,0)", "sum(2,1)"],
            ["sum(2,0)", "sum(2,1)", "sum(2,2)", "img(1)"],
        ]:
            with self.subTest(output_names=output_names):
                control, inputs = self.parse_addition(2, 2)
                outputs = self.output_literals(control, *output_names)
                # The first selection omits sum=2; the second also selects
                # an always-true fact, giving each model two output atoms.
                with self.assertRaisesRegex(ValueError, "exactly one selected output"):
                    ModuleGenerator(control).generate_module(inputs, outputs)

    def test_generate_module_rejects_nondefinite_programs(self) -> None:
        """Missing or multiple stable models for an input combination are rejected."""
        cases = [
            (
                ":- digit(1,1,0), digit(1,2,0).",
                "Expected 4 input combinations, found 3",
            ),
            ("{hidden}.\n#show sum/2.", "multiple stable models"),
        ]
        for extra_rules, message in cases:
            with self.subTest(extra_rules=extra_rules):
                program = (
                    self.NEURASP_ADDITION.format(images=2, domain="0,1") + extra_rules
                )
                control, inputs = self.parse_module(program)
                outputs = self.output_literals(
                    control, "sum(2,0)", "sum(2,1)", "sum(2,2)"
                )
                control.config.solve.models = "1"
                control.config.solve.project = "show"
                project_setting = control.config.solve.project.value

                # A constraint removes an input assignment; a hidden choice
                # gives two stable models per assignment. Both violate the
                # simple generator's definiteness assumption. Projection and
                # a model limit must not hide this evidence from compilation.
                with self.assertRaisesRegex(ValueError, message):
                    ModuleGenerator(control).generate_module(inputs, outputs)
                self.assertEqual(control.config.solve.models.value, "1")
                self.assertEqual(control.config.solve.project.value, project_setting)

    def test_duplicate_detection_with_unequal_group_sizes(self) -> None:
        """Every combination of groups of 2, 2 and 3 atoms gets a distinct code: the
        12 models compile, and a second model per combination is still caught.
        """
        program = """
            nn(digit(2,item), [0,1]).
            nn(offset(1,item), [0,1,2]).
            total(A+B+C) :- digit(1,item,A), digit(2,item,B), offset(1,item,C).
        """
        outputs = [f"total({value})" for value in range(5)]
        control, inputs = self.parse_module(program)
        models = ModuleGenerator(control).generate_models(
            inputs, self.output_literals(control, *outputs)
        )
        self.assertEqual(sum(len(rows) for rows in models.values()), 12)
        # Each row is one combination: one atom of each group, all different.
        rows = torch.cat(list(models.values())).tolist()
        self.assertEqual(len({tuple(row) for row in rows}), 12)
        control, inputs = self.parse_module(program + "{hidden}.")
        with self.assertRaisesRegex(ValueError, "multiple stable models"):
            ModuleGenerator(control).generate_models(
                inputs, self.output_literals(control, *outputs)
            )

    def test_generate_models_requires_one_atom_per_input_group(self) -> None:
        """A model with no atom of an input group, or two, is rejected."""
        for choice in ["{a; b}.", "{a}. b."]:
            with self.subTest(choice=choice):
                lib = Library()
                control = Control(lib, ["0"])
                control.parse_string(f"{choice}\nout.")
                control.ground()
                inputs = InputDescription.parse_inputs(lib, control, [["a", "b"]])
                output = control.base[("out", 0)][parse_symbol_term(lib, "out")]
                with self.assertRaisesRegex(
                    ValueError, "exactly one atom from each input group"
                ):
                    ModuleGenerator(control).generate_models(inputs, [output.literal])


if __name__ == "__main__":
    unittest.main()
