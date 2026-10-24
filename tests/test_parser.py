from typing import TYPE_CHECKING
import unittest

from neurasp.parser import (
    ParseError,
    bracket_neural_domains,
    parse_ground_neural_atom,
    parse_mvpp_choices,
    parse_neurasp_program,
    remove_neural_declaration_lines,
    split_top_level,
    symbols_from_control_base,
)

if TYPE_CHECKING:
    from clingo.control import Control
    from clingo.core import Library
    from clingo.symbol import Symbol


def require_clingo() -> "tuple[type[Library], type[Control]]":
    """Return Clingo 6's Library and Control classes, skipping the test if it is
    missing.
    """
    try:
        from clingo.control import Control
        from clingo.core import Library
    except ImportError as exc:
        raise unittest.SkipTest("clingo 6 is not installed") from exc
    return Library, Control


def ground_symbols(program: str) -> "list[Symbol]":
    """Ground program with Clingo and return its symbolic atoms."""
    Library, Control = require_clingo()
    control = Control(Library(), [])
    control.parse_string(program)
    control.ground()
    return symbols_from_control_base(control)


class ParserTests(unittest.TestCase):
    """Parsing NeurASP programs and their translation to standard ASP."""

    def test_split_top_level_respects_quotes_and_terms(self) -> None:
        """Commas inside quotes, function terms and tuples do not split."""
        self.assertEqual(
            split_top_level('a,f(1,2),"x,y",(b,c)'),
            ["a", "f(1,2)", '"x,y"', "(b,c)"],
        )

    def test_parse_ground_neural_atom(self) -> None:
        """A ground nn/2 atom gives its network, output count, term, domain, width."""
        symbols = ground_symbols('nn(label(1,img,b0),("car","person","truck")).')
        atom = parse_ground_neural_atom(
            next(symbol for symbol in symbols if symbol.name == "nn")
        )
        self.assertEqual(atom.network, "label")
        self.assertEqual(atom.output_count, 1)
        self.assertEqual(atom.term, "img,b0")
        self.assertEqual(atom.domain, ('"car"', '"person"', '"truck"'))
        self.assertEqual(atom.output_width, 3)

    def test_yolo_neural_atoms_are_rejected(self) -> None:
        """The original NeurASP's yolo output count is not supported."""
        symbols = ground_symbols("nn(label(yolo,img),(car,person)).")
        nn_atom = next(symbol for symbol in symbols if symbol.name == "nn")
        with self.assertRaisesRegex(ParseError, "yolo neural atoms"):
            parse_ground_neural_atom(nn_atom)

    def test_brackets_are_rewritten_only_in_neural_declarations(self) -> None:
        """Domains become tuples; weak constraint weights and strings are kept."""
        self.assertEqual(
            bracket_neural_domains('nn(d(1,X), ["a[",b]). :~ s. [1@1] q("[x]").'),
            'nn(d(1,X), ("a[",b)). :~ s. [1@1] q("[x]").',
        )

    def test_weak_constraints_are_kept(self) -> None:
        """Weak constraints, also split across lines, parse and stay in the program."""
        parsed = parse_neurasp_program("""
                img(i1).
                nn(digit(1,X), [0,1]) :- img(X).
                :~ digit(1,i1,V). [V@1]
                :~ digit(1,i1,0).
                   [1@2]
            """)
        self.assertIn(":~ digit(1,i1,V). [V@1]", parsed.standard_asp_program)
        self.assertIn(":~ digit(1,i1,0). [1@2]", parsed.standard_asp_program)

    def test_parse_mvpp_choices(self) -> None:
        """A learnable probabilistic rule becomes a choice rule; ASP lines are kept."""
        parsed = parse_mvpp_choices("@0.2 a(1); @0.8 b(1).\nc :- a(1).")
        self.assertEqual(parsed.parameters, ((0.2, 0.8),))
        self.assertEqual(parsed.pi_prime, "1{a(1); b(1)}1.\nc :- a(1).")
        self.assertEqual(str(parsed.probabilistic_rules[0]), "@0.2 a(1); @0.8 b(1).")

    def test_neural_output_slots_are_one_based(self) -> None:
        """An atom with 3 outputs grounds choices for slots 1, 2 and 3, not 0..2."""
        parsed = parse_neurasp_program("nn(label(3,item(1)), [off,on]).")
        self.assertEqual(parsed.neural_atoms[0].output_count, 3)
        symbols = ground_symbols(parsed.standard_asp_program)
        self.assertEqual(
            {str(symbol) for symbol in symbols if symbol.name == "label"},
            {
                "label(1,item(1),off)",
                "label(1,item(1),on)",
                "label(2,item(1),off)",
                "label(2,item(1),on)",
                "label(3,item(1),off)",
                "label(3,item(1),on)",
            },
        )

    def test_remove_neural_declaration_lines_uses_ast_filter(self) -> None:
        """Only the nn/2 declaration rule is removed; other rules are kept."""
        require_clingo()
        filtered = remove_neural_declaration_lines("""
                img(i1).
                nn(digit(1,X), [0,1,2]) :- img(X).
                ok :- img(i1).
            """)
        self.assertNotIn("nn(", filtered)
        self.assertIn("img(i1).", filtered)
        self.assertIn("ok :- img(i1).", filtered)

    def test_parse_neurasp_program_with_clingo_grounder_hook(self) -> None:
        """The grounder test hook is used, and nn rules become choice rules."""
        program = """
                    img(i1). img(i2).
                    nn(digit(1,X), [0,1,2]) :- img(X).
                    addition(A,B,N) :- digit(1,A,N1), digit(1,B,N2), N=N1+N2.
                """
        parsed = parse_neurasp_program(
            program,
            grounder=ground_symbols,
        )
        self.assertEqual(len(parsed.neural_atoms), 2)
        self.assertNotIn("@0.0", parsed.standard_asp_program)
        self.assertNotIn("nn(", parsed.standard_asp_program)
        self.assertIn("1{digit(1,i1,0);", parsed.standard_asp_program)

    def test_parse_neurasp_program_with_real_clingo_symbols(self) -> None:
        """With Clingo grounding, a quoted domain becomes a choice rule over its
        strings.
        """
        require_clingo()
        parsed = parse_neurasp_program("""
                img(i1).
                nn(label(1,X), ["cat","dog"]) :- img(X).
            """)

        self.assertEqual(len(parsed.neural_atoms), 1)
        self.assertEqual(parsed.neural_atoms[0].network, "label")
        self.assertEqual(parsed.neural_atoms[0].term, "i1")
        self.assertEqual(parsed.neural_atoms[0].domain, ('"cat"', '"dog"'))
        self.assertIn(
            '1{label(1,i1,"cat"); label(1,i1,"dog")}1.', parsed.standard_asp_program
        )

    def test_parse_neurasp_program_keeps_mvpp_rule_compatibility(self) -> None:
        """A program without neural atoms keeps its MVPP rule as a choice rule."""
        require_clingo()
        parsed = parse_neurasp_program("""
                @0.2 rain; @0.8 dry.
                cloudy :- rain.
            """)

        self.assertEqual(parsed.neural_atoms, tuple())
        self.assertIn("1{rain; dry}1.", parsed.standard_asp_program)
        self.assertIn("cloudy :- rain.", parsed.standard_asp_program)


if __name__ == "__main__":
    unittest.main()
