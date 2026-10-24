"""Check the compile cache: entries are reused while their key matches, and any
change to what determines the compiled program makes them stale."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import neurasp.cache
from neurasp.cache import (
    cache_path,
    cached_compile_program,
    compile_key,
    load_compiled,
    save_compiled,
)
from neurasp.compiler import CompiledProgram, EnumerationOptions, compile_program

PROGRAM = """
img(a). img(b).
nn(digit(1,X), [0,1,2]) :- img(X).
total(N) :- digit(1,a,A), digit(1,b,B), N=A+B.
"""
OUTPUTS = [f"total({value})" for value in range(5)]


class CompileCacheTests(unittest.TestCase):
    """cached_compile_program and the functions it is built from."""

    def assert_same_compiled(
        self, actual: CompiledProgram, expected: CompiledProgram
    ) -> None:
        """actual has expected's neural atoms and models, output by output."""
        self.assertEqual(actual.neural_atoms, expected.neural_atoms)
        self.assertEqual(list(actual.models), list(expected.models))
        for column, rows in expected.models.items():
            self.assertEqual(actual.models[column].dtype, torch.int32)
            torch.testing.assert_close(actual.models[column], rows)

    def test_round_trip(self) -> None:
        """A saved compiled program loads back with its key and not with another."""
        compiled = compile_program(PROGRAM, OUTPUTS)
        with tempfile.TemporaryDirectory() as directory:
            path = cache_path(Path(directory), "digits")
            self.assertIsNone(load_compiled(path, "key"))
            save_compiled(path, "key", compiled)
            loaded = load_compiled(path, "key")
            assert loaded is not None
            self.assert_same_compiled(loaded, compiled)
            self.assertIsNone(load_compiled(path, "other key"))
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["digits.pt"])

    def test_key_depends_on_program_and_outputs_only(self) -> None:
        """Another program or other (or reordered) outputs give another key;
        enumeration options are not part of it.
        """
        key = compile_key(PROGRAM, OUTPUTS)
        self.assertEqual(key, compile_key(PROGRAM, list(OUTPUTS)))
        self.assertNotEqual(key, compile_key(PROGRAM + "img(c).", OUTPUTS))
        self.assertNotEqual(key, compile_key(PROGRAM, OUTPUTS[::-1]))
        self.assertNotEqual(key, compile_key(PROGRAM, OUTPUTS[:-1]))

    def test_cached_compile_program(self) -> None:
        """The first call compiles and writes the entry, the next ones read it, also
        with other enumeration options; a changed key or rebuild compiles again, and
        no cache directory reads and writes nothing.
        """
        expected = compile_program(PROGRAM, OUTPUTS)
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)

            def run(
                options: EnumerationOptions = EnumerationOptions(),
                rebuild: bool = False,
            ) -> tuple[CompiledProgram, str]:
                """cached_compile_program on PROGRAM in cache_dir."""
                return cached_compile_program(
                    PROGRAM,
                    OUTPUTS,
                    options,
                    cache_dir=cache_dir,
                    name="digits",
                    rebuild=rebuild,
                )

            self.assertEqual(run()[1], "clingo")
            compiled, source = run(options=EnumerationOptions(processes=2))
            self.assertEqual(source, "cache")
            self.assert_same_compiled(compiled, expected)
            with patch.object(neurasp.cache, "compile_key", return_value="changed"):
                self.assertEqual(run()[1], "clingo")
                self.assertEqual(run()[1], "cache")
            self.assertEqual(run()[1], "clingo")
            self.assertEqual(run(rebuild=True)[1], "clingo")
            self.assertEqual(run()[1], "cache")
            self.assertEqual([p.name for p in cache_dir.iterdir()], ["digits.pt"])
        compiled, source = cached_compile_program(
            PROGRAM, OUTPUTS, cache_dir=None, name="digits"
        )
        self.assertEqual(source, "clingo")
        self.assert_same_compiled(compiled, expected)


if __name__ == "__main__":
    unittest.main()
