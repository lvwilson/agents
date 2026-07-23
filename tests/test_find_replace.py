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


if __name__ == "__main__":
    unittest.main()
