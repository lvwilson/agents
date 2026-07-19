import unittest
import difflib
from agents.tools.codemanipulator import (create_code, replace_code, remove_code, format_code,
    syntax_check, insert_code_before, insert_code_after,
    get_signatures_and_docstrings, read_code_at_address, change_docstring,
    convert_double_quotes_to_single)

class TestCodeManipulator(unittest.TestCase):
    def setUp(self):
        self.source_code = format_code(
            '''
"""
This is a module docstring.
"""

class MathUtils:
    """
    Math utilities class.
    """

    @staticmethod
    def factorial(n):
        """
        Calculate the factorial of a number.
        """
        if n == 0:
            return 1
        else:
            return n * MathUtils.factorial(n - 1)

    @staticmethod
    def multiple(a, b):
        """
        Multiply two numbers.
        """
        return a * b

    class InnerClass:
        """
        An inner class.
        """
        def inner_method(self):
            """
            An inner method.
            """
            return "inner result"

def standalone_function(x):
    """
    Increment a number.
    """
    return x + 1

def another_function(y):
    """
    Decrement a number.
    """
    return y - 1
'''
        )

    def assertCodeEqual(self, actual, expected):
        actual_formatted = format_code(actual).strip()
        expected_formatted = format_code(expected).strip()
        if actual_formatted != expected_formatted:
            diff = difflib.unified_diff(
                expected_formatted.splitlines(),
                actual_formatted.splitlines(),
                fromfile="expected",
                tofile="actual",
                lineterm=""
            )
            diff_text = "\n".join(diff)
            self.fail(f"Code does not match:\n{diff_text}")

    def test_replace_function(self):
        address = "MathUtils.factorial"
        new_code = '''
@staticmethod
def factorial(n):
    """
    Calculate the factorial of a number.
    Optimized version.
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
'''
        expected_code = format_code(
            '''
"""
This is a module docstring.
"""

class MathUtils:
    """
    Math utilities class.
    """

    @staticmethod
    def factorial(n):
        """
        Calculate the factorial of a number.
        Optimized version.
        """
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @staticmethod
    def multiple(a, b):
        """
        Multiply two numbers.
        """
        return a * b

    class InnerClass:
        """
        An inner class.
        """
        def inner_method(self):
            """
            An inner method.
            """
            return "inner result"

def standalone_function(x):
    """
    Increment a number.
    """
    return x + 1

def another_function(y):
    """
    Decrement a number.
    """
    return y - 1
'''
        )
        result_code = replace_code(self.source_code, address, new_code)
        self.assertCodeEqual(result_code, expected_code)

    def test_create_function(self):
        address = "MathUtils.divide"
        new_code = '''
@staticmethod
def divide(a, b):
    """
    Divide two numbers.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
        expected_code = format_code(
            '''
"""
This is a module docstring.
"""

class MathUtils:
    """
    Math utilities class.
    """

    @staticmethod
    def factorial(n):
        """
        Calculate the factorial of a number.
        """
        if n == 0:
            return 1
        else:
            return n * MathUtils.factorial(n - 1)

    @staticmethod
    def multiple(a, b):
        """
        Multiply two numbers.
        """
        return a * b

    class InnerClass:
        """
        An inner class.
        """
        def inner_method(self):
            """
            An inner method.
            """
            return "inner result"
    
    @staticmethod
    def divide(a, b):
        """
        Divide two numbers.
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

def standalone_function(x):
    """
    Increment a number.
    """
    return x + 1

def another_function(y):
    """
    Decrement a number.
    """
    return y - 1
'''
        )
        result_code = create_code(self.source_code, address, new_code)
        self.assertCodeEqual(result_code, expected_code)

    def test_remove_function(self):
        address = "MathUtils.multiple"
        expected_code = format_code(
            '''
"""
This is a module docstring.
"""

class MathUtils:
    """
    Math utilities class.
    """

    @staticmethod
    def factorial(n):
        """
        Calculate the factorial of a number.
        """
        if n == 0:
            return 1
        else:
            return n * MathUtils.factorial(n - 1)

    class InnerClass:
        """
        An inner class.
        """
        def inner_method(self):
            """
            An inner method.
            """
            return "inner result"

def standalone_function(x):
    """
    Increment a number.
    """
    return x + 1

def another_function(y):
    """
    Decrement a number.
    """
    return y - 1
'''
        )
        result_code = remove_code(self.source_code, address)
        self.assertCodeEqual(result_code, expected_code)

    def test_replace_non_existing_function(self):
        address = "MathUtils.non_existing_method"
        new_code = '''
@staticmethod
def non_existing_method():
    """
    This should raise an error because the method doesn't exist.
    """
    pass
'''
        with self.assertRaises(ValueError):
            replace_code(self.source_code, address, new_code)

    def test_remove_non_existing_function(self):
        address = "MathUtils.non_existing_method"
        with self.assertRaises(ValueError):
            remove_code(self.source_code, address)

    def test_replace_class(self):
        source_code = """
class TestClass:
    \"\"\"
    A test class that stores a value and can double it.

    Attributes:
    value (int): The stored value.
    \"\"\"

    def __init__(self, value):
        self.value = value

    def double_value(self):
        return self.value * 4


def main():
    test_instance = TestClass(5)
    result = test_instance.double_value()
    print(f"The doubled value is: {result}")


if __name__ == "__main__":
    main()"""
        
        new_class = """
class TestClass:
    \"\"\"
    A test class that stores a value and can double it.

    Attributes:
    value (int): The stored value.
    \"\"\"

    def __init__(self, value):
        self.value = value

    def double_value(self):
        return self.value * 2"""

        expected_code = """
class TestClass:
    \"\"\"
    A test class that stores a value and can double it.

    Attributes:
    value (int): The stored value.
    \"\"\"

    def __init__(self, value):
        self.value = value

    def double_value(self):
        return self.value * 2


def main():
    test_instance = TestClass(5)
    result = test_instance.double_value()
    print(f"The doubled value is: {result}")


if __name__ == "__main__":
    main()"""

        result_code = replace_code(source_code, "TestClass", new_class)
        self.assertCodeEqual(result_code, expected_code)


class TestManipulatorUtilities(unittest.TestCase):
    """Direct coverage for the module-level helpers that had none."""

    SIMPLE = (
        "def foo():\n"
        "    return 1\n"
        "\n"
        "def bar():\n"
        "    return 2\n"
    )

    CLASS = (
        "class Greeter:\n"
        "    \"\"\"Greeting utilities.\"\"\"\n"
        "    def hello(self):\n"
        "        \"\"\"Say hello.\"\"\"\n"
        "        return 'hi'\n"
    )

    def test_syntax_check_valid_and_invalid(self):
        self.assertTrue(syntax_check("x = 1\n"))
        self.assertFalse(syntax_check("def broken(:\n"))

    def test_insert_code_before_module_function(self):
        new_func = "def inserted():\n    return 0\n"
        result = insert_code_before(self.SIMPLE, "bar", new_func)
        self.assertLess(result.index("def inserted"), result.index("def bar"))
        self.assertIn("def foo", result)

    def test_insert_code_after_module_function(self):
        new_func = "def inserted():\n    return 0\n"
        result = insert_code_after(self.SIMPLE, "foo", new_func)
        self.assertLess(result.index("def foo"), result.index("def inserted"))
        self.assertLess(result.index("def inserted"), result.index("def bar"))

    def test_get_signatures_and_docstrings(self):
        result = get_signatures_and_docstrings(self.CLASS)
        self.assertIn("class Greeter:", result)
        self.assertIn("def hello(self):", result)
        self.assertIn("Say hello.", result)

    def test_read_code_at_address(self):
        result = read_code_at_address(self.CLASS, "Greeter.hello")
        self.assertIn("def hello(self):", result)
        self.assertIn("return 'hi'", result)
        # Unknown address reports rather than raising.
        self.assertIn("No code found", read_code_at_address(self.CLASS, "Greeter.nope"))

    def test_change_docstring(self):
        result = change_docstring(self.CLASS, "Greeter.hello", '"Documented afresh."')
        self.assertIn("Documented afresh.", result)
        self.assertNotIn("Say hello.", result)
        with self.assertRaises(ValueError):
            change_docstring(self.CLASS, "Greeter.missing", '"x"')

    def test_convert_double_quotes_to_single(self):
        self.assertEqual(convert_double_quotes_to_single('x = "hello"'), "x = 'hello'")

    def test_remove_async_function(self):
        source = (
            "async def afunc():\n"
            "    return 1\n"
            "\n"
            "def keep():\n"
            "    return 2\n"
        )
        result = remove_code(source, "afunc")
        self.assertNotIn("afunc", result)
        self.assertIn("def keep", result)


if __name__ == "__main__":
    unittest.main()