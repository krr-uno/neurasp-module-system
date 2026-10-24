"""Check compile_program's input and output column order and its output checks."""

import itertools
import unittest

import torch

from neurasp.compiler import EnumerationOptions, compile_program
from neurasp.module import NeurASPModule

# Atoms declared out of order (b before a), plus one atom with two output slots.
PROGRAM = """
img(b). img(a).
nn(digit(1,X), [0,1]) :- img(X).
nn(pair(2,c), [0,1,2]).
total(N) :- digit(1,a,A), digit(1,b,B), pair(1,c,C), pair(2,c,D), N=A+B+C+D.
"""
OUTPUTS = [f"total({value})" for value in range(7)]


class CompileProgramTests(unittest.TestCase):
    """compile_program's column order and argument checks."""

    def test_input_columns_follow_terms_slots_and_domains(self) -> None:
        """Input columns follow atoms sorted by term, then slots, then domain values;
        every combination gives its sum.
        """
        compiled = compile_program(PROGRAM, OUTPUTS)
        self.assertEqual([atom.term for atom in compiled.neural_atoms], ["a", "b", "c"])
        module = NeurASPModule(compiled.models)
        self.assertEqual(len(module.models), 2 * 2 * 3 * 3)
        # Columns: digit a (2), digit b (2), pair slot 1 (3), pair slot 2 (3).
        sizes = [2, 2, 3, 3]
        for values in itertools.product(*(range(size) for size in sizes)):
            inputs = torch.cat(
                [
                    torch.nn.functional.one_hot(torch.tensor([value]), size)
                    for value, size in zip(values, sizes)
                ],
                dim=1,
            ).float()
            expected = torch.nn.functional.one_hot(torch.tensor([sum(values)]), 7)
            torch.testing.assert_close(module(inputs), expected.float())

    def test_terms_are_in_natural_order(self) -> None:
        """Digits in terms compare as numbers: i2 comes before i10."""
        program = """
        img(i10;i2;i1).
        nn(digit(1,X), [0,1]) :- img(X).
        total(A+B+C) :- digit(1,i1,A), digit(1,i2,B), digit(1,i10,C).
        """
        compiled = compile_program(program, [f"total({n})" for n in range(4)])
        self.assertEqual(
            [atom.term for atom in compiled.neural_atoms], ["i1", "i2", "i10"]
        )

    def test_programs_own_show_directives_do_not_hide_inputs(self) -> None:
        """A program that shows other atoms compiles to the same models."""
        expected = compile_program(PROGRAM, OUTPUTS).models
        shown = compile_program(PROGRAM + "#show img/1.", OUTPUTS).models
        self.assertEqual(
            {column: [m.tolist() for m in models] for column, models in shown.items()},
            {
                column: [m.tolist() for m in models]
                for column, models in expected.items()
            },
        )

    def test_outputs_must_be_atoms(self) -> None:
        """A number or a string as an output atom is a ValueError, not an assert."""
        for bad in ["5", '"x"']:
            with self.subTest(output=bad):
                with self.assertRaisesRegex(ValueError, "Expected an atom"):
                    compile_program(PROGRAM, [*OUTPUTS[:-1], bad])

    def test_output_columns_follow_the_given_atoms(self) -> None:
        """Output column n is the probability of outputs[n], in the order given."""
        module = NeurASPModule(compile_program(PROGRAM, OUTPUTS[::-1]).models)
        # digit a = 1, digit b = 0, pair slots = 0 and 0: total 1, so column 6 - 1 = 5
        # with the outputs reversed (column 1 in the given order).
        inputs = torch.tensor([[0.0, 1, 1, 0, 1, 0, 0, 1, 0, 0]])
        torch.testing.assert_close(
            module(inputs), torch.nn.functional.one_hot(torch.tensor([5]), 7).float()
        )

    def test_output_atoms_must_be_grounded(self) -> None:
        """An output atom that cannot be derived from the program is rejected."""
        with self.assertRaisesRegex(ValueError, "total.7. cannot be derived"):
            compile_program(PROGRAM, [*OUTPUTS, "total(7)"])

    def test_every_stable_model_needs_a_listed_output(self) -> None:
        """Leaving out a possible output leaves some stable models without one."""
        with self.assertRaisesRegex(ValueError, "exactly one selected output"):
            compile_program(PROGRAM, OUTPUTS[:-1])


def sorted_models(models: dict[int, torch.Tensor]) -> dict[int, list[list[int]]]:
    """Each output's models as sorted rows, to compare enumerations in any order."""
    return {column: sorted(block.tolist()) for column, block in models.items()}


class EnumerationOptionsTests(unittest.TestCase):
    """compile_program enumerates the same models under every EnumerationOptions."""

    def test_options_give_the_same_models(self) -> None:
        """Processes, Clingo threads and enumeration modes change only the order."""
        expected = sorted_models(compile_program(PROGRAM, OUTPUTS).models)
        for options in [
            EnumerationOptions(processes=2),
            EnumerationOptions(processes=3),
            EnumerationOptions(threads=2, parallel_mode="split"),
            EnumerationOptions(threads=3),
            EnumerationOptions(enum_mode="record"),
            EnumerationOptions(enum_mode="bt", processes=2, threads=2),
        ]:
            with self.subTest(options=options):
                models = compile_program(PROGRAM, OUTPUTS, options).models
                self.assertEqual(sorted_models(models), expected)

    def test_processes_report_invalid_programs(self) -> None:
        """With several processes, a model without a listed output and a missing
        combination are still ValueErrors.
        """
        options = EnumerationOptions(processes=2)
        with self.assertRaisesRegex(ValueError, "exactly one selected output"):
            compile_program(PROGRAM, OUTPUTS[:-1], options)
        program = PROGRAM + ":- digit(1,a,1), digit(1,b,1).\n"
        with self.assertRaisesRegex(ValueError, "Expected 36 input combinations"):
            compile_program(program, OUTPUTS, options)

    def test_invalid_options(self) -> None:
        """Nonpositive counts and unknown modes are ValueErrors."""
        for options in [
            EnumerationOptions(threads=0),
            EnumerationOptions(processes=0),
            EnumerationOptions(parallel_mode="share"),
            EnumerationOptions(enum_mode="brave"),
        ]:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    compile_program(PROGRAM, OUTPUTS, options)


if __name__ == "__main__":
    unittest.main()
