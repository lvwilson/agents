import unittest
from agents.tools.findreplace import find_replace


class TestFindReplace(unittest.TestCase):

    def setUp(self):
        self.source_code = '''\
class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def start_engine(self):
        print(f"{self.brand} {self.model} engine started.")

def complex_calculation(x):
    def inner_calculation(y):
        return x * y + (lambda z: z**2)(y)
    return inner_calculation

class Container:
    class Inner:
        def __init__(self, value):
            self.value = value
'''

    def test_simple_replacement_in_class_method(self):
        command = """<<<<<<< SEARCH
    def start_engine(self):
        print(f"{self.brand} {self.model} engine started.")
=======
    def start_engine(self):
        print(f"{self.brand} {self.model} roars to life!")
>>>>>>> REPLACE"""
        expected_output_contains = 'print(f"{self.brand} {self.model} roars to life!")'
        result = find_replace(self.source_code, command)
        self.assertIn(expected_output_contains, result)

    def test_replace_nested_function_in_decorated_function(self):
        command = """<<<<<<< SEARCH
    def inner_calculation(y):
        return x * y + (lambda z: z**2)(y)
=======
    def inner_calculation(y):
        return x * y + (lambda z: z**3)(y)
>>>>>>> REPLACE"""
        expected_output_contains = "return x * y + (lambda z: z**3)(y)"
        result = find_replace(self.source_code, command)
        self.assertIn(expected_output_contains, result)

    def test_replace_class_variable_in_nested_class(self):
        command = """<<<<<<< SEARCH
        self.value = value
=======
        self.value = value * 2
>>>>>>> REPLACE"""
        expected_output_contains = "self.value = value * 2"
        result = find_replace(self.source_code, command)
        self.assertIn(expected_output_contains, result)

    def test_no_match_raises_error(self):
        command = """<<<<<<< SEARCH
def nonexistent_function():
    pass
=======
def replacement():
    pass
>>>>>>> REPLACE"""
        # A SEARCH text that matches nothing must raise, not silently no-op.
        with self.assertRaises(ValueError):
            find_replace(self.source_code, command)

    def test_invalid_command_format(self):
        command = "This is not a valid find-replace command"
        with self.assertRaises(ValueError):
            find_replace(self.source_code, command)

    def test_duplicate_match_raises_error(self):
        """A SEARCH text occurring more than once must raise (all-or-nothing)."""
        command = """<<<<<<< SEARCH
    pass
=======
    return None
>>>>>>> REPLACE"""
        source = "def a():\n    pass\n\ndef b():\n    pass\n"
        with self.assertRaises(ValueError):
            find_replace(source, command)
        # Original string untouched (all-or-nothing semantics).
        self.assertEqual(source, "def a():\n    pass\n\ndef b():\n    pass\n")


class TestSnapshotReplacement(unittest.TestCase):
    @staticmethod
    def apply(source, edits):
        from agents.tools.findreplace import validated_find_replace
        command = "\n".join(
            f"<<<<<<< SEARCH\n{search}\n=======\n{replacement}\n>>>>>>> REPLACE"
            for search, replacement in edits
        )
        return validated_find_replace(source, command)

    def test_length_changes_in_any_command_order(self):
        import itertools
        edits = [("alpha", "a much longer first line"), ("beta", ""), ("gamma", "g")]
        for order in itertools.permutations(edits):
            with self.subTest(order=order):
                result = self.apply("alpha\nbeta\ngamma\n", order)
                self.assertEqual(result.content, "a much longer first line\n\ng\n")
                self.assertEqual(result.applied_blocks, 3)
                self.assertEqual(result.errors, [])

    def test_replacement_cannot_create_ambiguity_for_another_block(self):
        result = self.apply("first\nsecond\n", [("first", "second"), ("second", "done")])
        self.assertEqual(result.content, "second\ndone\n")
        self.assertEqual(result.errors, [])

    def test_originally_ambiguous_search_stays_invalid(self):
        result = self.apply("x\nx\ny\n", [("x\nx", "z"), ("x", "bad"), ("y", "good")])
        self.assertEqual(result.content, "z\ngood\n")
        self.assertEqual(result.applied_blocks, 2)
        self.assertIn("Block 2: SEARCH text matches 2 locations", result.errors[0])

    def test_overlapping_occurrences_are_ambiguous(self):
        result = self.apply("banana", [("ana", "X")])
        self.assertEqual(result.content, "banana")
        self.assertIn("matches 2 locations", result.errors[0])

    def test_empty_search_is_rejected_even_for_empty_source(self):
        result = self.apply("", [("", "inserted")])
        self.assertEqual(result.content, "")
        self.assertEqual(result.applied_blocks, 0)
        self.assertIn("SEARCH text is empty", result.errors[0])

    def test_adjacent_ranges_are_not_conflicts(self):
        result = self.apply("abcdef", [("abc", "A"), ("def", "D")])
        self.assertEqual(result.content, "AD")
        self.assertEqual(result.errors, [])

    def test_identical_ranges_are_both_rejected(self):
        result = self.apply("abc", [("abc", "A"), ("abc", "B")])
        self.assertEqual(result.content, "abc")
        self.assertEqual(result.applied_blocks, 0)
        self.assertEqual(len(result.errors), 2)

    def test_every_block_in_overlap_chain_is_rejected(self):
        result = self.apply(
            "abcdefghij\nsafe", [("abcd", "A"), ("defg", "B"), ("ghij", "C"), ("safe", "OK")]
        )
        self.assertEqual(result.content, "abcdefghij\nOK")
        self.assertEqual(result.applied_blocks, 1)
        self.assertEqual(len(result.errors), 3)

    def test_nested_ranges_are_both_rejected(self):
        result = self.apply("abcdef", [("abcdef", "outer"), ("cd", "inner")])
        self.assertEqual(result.content, "abcdef")
        self.assertEqual(len(result.errors), 2)

    def test_all_errors_are_reported_in_block_order(self):
        result = self.apply(
            "A\nrepeat\nrepeat\n",
            [("missing", "X"), ("A\nrepeat", "Y"), ("A", "Z"), ("repeat", "Q")],
        )
        self.assertEqual(result.applied_blocks, 0)
        self.assertEqual(len(result.errors), 4)
        for number, error in enumerate(result.errors, 1):
            self.assertTrue(error.startswith(f"Block {number}:"))

    def test_strict_api_reports_all_errors(self):
        command = (
            "<<<<<<< SEARCH\nmissing1\n=======\nx\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\nmissing2\n=======\ny\n>>>>>>> REPLACE"
        )
        with self.assertRaises(ValueError) as caught:
            find_replace("original", command)
        self.assertIn("Block 1:", str(caught.exception))
        self.assertIn("Block 2:", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
