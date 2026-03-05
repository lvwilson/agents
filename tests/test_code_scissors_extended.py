import unittest
from agents.tools.code_scissors import insert_before, insert_after, replace_before, replace_after, insert_between, replace_between

class TestCodeScissorsExtended(unittest.TestCase):
    def setUp(self):
        self.complex_code = '''
import logging
from functools import wraps

def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

class ComplexClass:
    """A complex class with multiple methods and nested structures."""

    def __init__(self, value):
        self.value = value

    @log_decorator
    def complex_method(self):
        """This is a complex method with a docstring."""
        def nested_function():
            return self.value * 2

        with open('test.txt', 'w') as f:
            f.write(str(nested_function()))

        return nested_function()

    @staticmethod
    def static_method():
        return "I'm a static method"

# Global variable
GLOBAL_CONSTANT = 42

def main():
    obj = ComplexClass(21)
    result = obj.complex_method()
    print(f"Result: {result}")
    print(f"Static method: {ComplexClass.static_method()}")
    print(f"Global constant: {GLOBAL_CONSTANT}")

if __name__ == '__main__':
    main()
'''

    def test_prepend_import(self):
        """Test prepending an import statement."""
        new_code = "from typing import List, Dict\n"
        result = insert_before(self.complex_code, "import logging", new_code)
        self.assertIn("from typing import List, Dict", result)
        self.assertIn("import logging", result)
        self.assertTrue(result.index("from typing import List, Dict") < result.index("import logging"))

    def test_append_method(self):
        """Test appending a new method to a class."""
        new_method = '''
    def new_method(self):
        """A new method added to the class."""
        return self.value + 10
'''
        result = insert_after(self.complex_code, "return \"I'm a static method\"", new_method)
        self.assertIn("def new_method(self):", result)
        self.assertIn("\"\"\"A new method added to the class.\"\"\"", result)
        self.assertIn("return self.value + 10", result)

    def test_replace_before_class(self):
        """Test replacing code before a class definition."""
        new_code = "# New imports and decorators\nfrom datetime import datetime\n\n"
        result = replace_before(self.complex_code, "class ComplexClass:", new_code)
        self.assertIn("from datetime import datetime", result)
        self.assertNotIn("import logging", result)
        self.assertIn("class ComplexClass:", result)

    def test_replace_after_method(self):
        """Test replacing code after a method definition."""
        new_code = "        return self.value * 3\n"
        result = replace_after(self.complex_code, "def complex_method(self):", new_code)
        self.assertIn("return self.value * 3", result)
        self.assertNotIn("def nested_function():", result)
        self.assertIn("@log_decorator", result)

    def test_insert_between_nested(self):
        """Test inserting code between nested structures."""
        new_code = "            print('Nested function called')\n"
        result = insert_between(self.complex_code, "def nested_function():", "return self.value * 2", new_code)
        self.assertIn("print('Nested function called')", result)
        self.assertIn("def nested_function():", result)
        self.assertIn("return self.value * 2", result)

    def test_replace_between_methods(self):
        """Test replacing code between two methods in a class."""
        new_code = '''
    def new_instance_method(self):
        return self.value + 5
'''
        result = replace_between(self.complex_code, "return nested_function()", "@staticmethod", new_code)
        # Both boundary lines are preserved
        self.assertIn("return nested_function()", result)
        self.assertIn("@staticmethod", result)
        # New code is inserted between the boundaries
        self.assertIn("def new_instance_method(self):", result)
        self.assertIn("return self.value + 5", result)
        # The new method appears between the two markers
        self.assertTrue(
            result.index("return nested_function()") <
            result.index("new_instance_method") <
            result.index("@staticmethod")
        )

    def test_multiple_operations(self):
        """Test applying multiple operations in sequence."""
        code = self.complex_code
        code = insert_before(code, "import logging", "import sys\n")
        code = insert_after(code, "class ComplexClass:", "\n    class_variable = 100\n")
        code = insert_before(code, "def main():", "# Main function below\n")
        code = replace_after(code, "if __name__ == '__main__':", "    logging.basicConfig(level=logging.INFO)\n    main()\n")
        
        self.assertIn("import sys", code)
        self.assertIn("class_variable = 100", code)
        self.assertIn("# Main function below", code)
        self.assertIn("logging.basicConfig(level=logging.INFO)", code)

    def test_prepend_error(self):
        """Test prepend function with non-existent cutting point."""
        with self.assertRaises(ValueError):
            insert_before(self.complex_code, "non_existent_line", "new_code")

    def test_append_error(self):
        """Test append function with non-existent cutting point."""
        with self.assertRaises(ValueError):
            insert_after(self.complex_code, "non_existent_line", "new_code")

    def test_replace_before_error(self):
        """Test replace_before function with non-existent cutting point."""
        with self.assertRaises(ValueError):
            replace_before(self.complex_code, "non_existent_line", "new_code")

    def test_replace_after_error(self):
        """Test replace_after function with non-existent cutting point."""
        with self.assertRaises(ValueError):
            replace_after(self.complex_code, "non_existent_line", "new_code")

    def test_insert_between_error(self):
        """Test insert_between function with non-existent cutting points."""
        with self.assertRaises(ValueError):
            insert_between(self.complex_code, "non_existent_line1", "non_existent_line2", "new_code")

    def test_replace_between_error(self):
        """Test replace_between function with non-existent cutting points."""
        with self.assertRaises(ValueError):
            replace_between(self.complex_code, "non_existent_line1", "non_existent_line2", "new_code")

    def test_empty_input(self):
        """Test functions with empty input."""
        empty_code = ""
        new_code = "print('Hello')\n"
        self.assertEqual(insert_before(empty_code, "any_line", new_code), new_code)
        with self.assertRaises(ValueError):
            insert_after(empty_code, "any_line", new_code)

    def test_cutting_point_at_beginning(self):
        """Test prepend with cutting point at the beginning of the code."""
        new_code = "# New beginning\n"
        result = insert_before(self.complex_code, "import logging", new_code)
        self.assertTrue(result.lstrip().startswith("# New beginning"))

    def test_cutting_point_at_end(self):
        """Test append with cutting point at the end of the code."""
        new_code = "# New end\n"
        result = insert_after(self.complex_code, "    main()", new_code)
        self.assertTrue(result.rstrip().endswith("# New end"))

    def test_newline_handling(self):
        """Test consistent newline handling."""
        new_code = "new_line_without_newline"
        result = insert_after(self.complex_code, "import logging", new_code)
        self.assertIn("new_line_without_newline\n", result)

if __name__ == '__main__':
    unittest.main()
