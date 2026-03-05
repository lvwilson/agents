import unittest
import difflib
from agents.tools.codemanipulator import create_code, replace_code, remove_code, format_code

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
if __name__ == "__main__":
    unittest.main()