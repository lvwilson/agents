import unittest
from agents.tools.code_scissors import insert_before, insert_after, replace_before, replace_after, insert_between, replace_between

class TestCodeScissors(unittest.TestCase):
    def setUp(self):
        self.sample_code = """def hello_world():
    print("Hello, World!")
    return None

def goodbye_world():
    print("Goodbye, World!")
    return None
"""

    def test_insert_before(self):
        """Test prepending code before a specific line."""
        new_code = "# This is a new comment\n"
        result = insert_before(self.sample_code, "def hello_world():", new_code)
        self.assertIn("# This is a new comment", result)
        self.assertIn("def hello_world():", result)
        self.assertTrue(result.index("# This is a new comment") < result.index("def hello_world():"))

    def test_insert_after(self):
        """Test appending code after a specific line."""
        new_code = "    print('Additional line')\n"
        result = insert_after(self.sample_code, "print(\"Hello, World!\")", new_code)
        self.assertIn("print(\"Hello, World!\")", result)
        self.assertIn("print('Additional line')", result)
        self.assertTrue(result.index("print(\"Hello, World!\")") < result.index("print('Additional line')"))

    def test_replace_before(self):
        """Test replacing code before a specific line."""
        new_code = "# New function\ndef new_function():\n    pass\n\n"
        result = replace_before(self.sample_code, "def goodbye_world():", new_code)
        self.assertIn("# New function", result)
        self.assertIn("def new_function():", result)
        self.assertIn("def goodbye_world():", result)
        self.assertNotIn("def hello_world():", result)

    def test_replace_after(self):
        """Test replacing code after a specific line."""
        new_code = "    print('New goodbye message')\n    return 'Goodbye'\n"
        result = replace_after(self.sample_code, "def goodbye_world():", new_code)
        self.assertIn("def goodbye_world():", result)
        self.assertIn("print('New goodbye message')", result)
        self.assertIn("return 'Goodbye'", result)
        self.assertNotIn("print(\"Goodbye, World!\")", result)

    def test_insert_between(self):
        """Test inserting code between two specific lines."""
        new_code = "def middle_function():\n    print('Middle')\n"
        result = insert_between(self.sample_code, "    return None", "def goodbye_world():", new_code)
        self.assertIn("def middle_function():", result)
        self.assertIn("print('Middle')", result)
        self.assertTrue(result.index("return None") < result.index("def middle_function():") < result.index("def goodbye_world():"))

    def test_replace_between(self):
        """Test replacing code between two specific lines."""
        new_code = "def new_function():\n    print('New function')\n"
        result = replace_between(self.sample_code, "def hello_world():", "def goodbye_world():", new_code)
        self.assertIn("def hello_world():", result)
        self.assertIn("def new_function():", result)
        self.assertIn("print('New function')", result)
        self.assertIn("def goodbye_world():", result)
        self.assertNotIn("print(\"Hello, World!\")", result)

    def test_error_on_non_existent_cutting_point(self):
        """All six functions raise ValueError when cutting point is not found."""
        for func, args in [
            (insert_before, ("non_existent_line", "new_code")),
            (insert_after, ("non_existent_line", "new_code")),
            (replace_before, ("non_existent_line", "new_code")),
            (replace_after, ("non_existent_line", "new_code")),
            (insert_between, ("non_existent_line1", "non_existent_line2", "new_code")),
            (replace_between, ("non_existent_line1", "non_existent_line2", "new_code")),
        ]:
            with self.assertRaises(ValueError):
                func(self.sample_code, *args)

    def test_empty_input(self):
        """Test functions with empty input."""
        empty_code = ""
        new_code = "print('Hello')\n"
        self.assertEqual(insert_before(empty_code, "any_line", new_code), new_code)
        with self.assertRaises(ValueError):
            insert_after(empty_code, "any_line", new_code)

    def test_newline_handling(self):
        """Test consistent newline handling."""
        new_code = "new_line_without_newline"
        result = insert_after(self.sample_code, "print(\"Hello, World!\")", new_code)
        self.assertIn("new_line_without_newline\n", result)

if __name__ == '__main__':
    unittest.main()
