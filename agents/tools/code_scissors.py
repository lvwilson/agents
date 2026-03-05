def insert_before(code, cutting_point, new_code):
    """
    Insert new_code before the line that matches the cutting_point.

    Args:
    code (str): The original code.
    cutting_point (str): The line before which to prepend the new code.
    new_code (str): The code to be prepended.

    Returns:
    str: The modified code with new_code prepended.

    Raises:
    ValueError: If the cutting_point is not found in the code.

    Note:
    - This function reads the entire input into memory, which may be inefficient for very large files.
    - The matching is done using string strip() method, which may be sensitive to whitespace differences.
    """
    lines = code.splitlines(True)
    if not lines:
        return new_code + code
    for i, line in enumerate(lines):
        if line.strip() == cutting_point.strip():
            return ''.join(lines[:i] + [new_code if new_code.endswith('\n') else new_code + '\n'] + lines[i:])
    raise ValueError(f"Cutting point '{cutting_point}' not found in the code.")

def insert_after(code, cutting_point, new_code):
    """
    Insert new_code after the line that matches the cutting_point.

    Args:
    code (str): The original code.
    cutting_point (str): The line after which to append the new code.
    new_code (str): The code to be appended.

    Returns:
    str: The modified code with new_code appended.

    Raises:
    ValueError: If the cutting_point is not found in the code.

    Note:
    - This function reads the entire input into memory, which may be inefficient for very large files.
    - The matching is done using string strip() method, which may be sensitive to whitespace differences.
    """
    lines = code.splitlines(True)
    for i, line in enumerate(lines):
        if line.strip() == cutting_point.strip():
            if i == len(lines) - 1:  # If cutting point is the last line
                return code.rstrip('\n') + '\n' + new_code
            return ''.join(lines[:i+1] + [new_code if new_code.endswith('\n') else new_code + '\n'] + lines[i+1:])
    raise ValueError(f"Cutting point '{cutting_point}' not found in the code.")

def replace_before(code, cutting_point, new_code):
    """
    Replace all code before the line that matches the cutting_point with new_code.

    Args:
    code (str): The original code.
    cutting_point (str): The line before which to replace the code.
    new_code (str): The code to replace the existing code.

    Returns:
    str: The modified code with the section before cutting_point replaced.

    Raises:
    ValueError: If the cutting_point is not found in the code.

    Note:
    - This function reads the entire input into memory, which may be inefficient for very large files.
    - The matching is done using string strip() method, which may be sensitive to whitespace differences.
    """
    lines = code.splitlines(True)
    for i, line in enumerate(lines):
        if line.strip() == cutting_point.strip():
            return (new_code if new_code.endswith('\n') else new_code + '\n') + ''.join(lines[i:])
    raise ValueError(f"Cutting point '{cutting_point}' not found in the code.")

def replace_after(code, cutting_point, new_code):
    """
    Replace all code after the line that matches the cutting_point with new_code.

    Args:
    code (str): The original code.
    cutting_point (str): The line after which to replace the code.
    new_code (str): The code to replace the existing code.

    Returns:
    str: The modified code with the section after cutting_point replaced.

    Raises:
    ValueError: If the cutting_point is not found in the code.

    Note:
    - This function reads the entire input into memory, which may be inefficient for very large files.
    - The matching is done using string strip() method, which may be sensitive to whitespace differences.
    """
    lines = code.splitlines(True)
    for i, line in enumerate(lines):
        if line.strip() == cutting_point.strip():
            return ''.join(lines[:i+1]) + (new_code if new_code.startswith('\n') else '\n' + new_code)
    raise ValueError(f"Cutting point '{cutting_point}' not found in the code.")

def insert_between(code, cutting_point1, cutting_point2, new_code):
    """
    Insert new_code between the lines that match cutting_point1 and cutting_point2.

    Args:
    code (str): The original code.
    cutting_point1 (str): The line after which to start inserting.
    cutting_point2 (str): The line before which to stop inserting.
    new_code (str): The code to be inserted.

    Returns:
    str: The modified code with new_code inserted between cutting_point1 and cutting_point2.

    Raises:
    ValueError: If either cutting_point1 or cutting_point2 is not found in the code.

    Note:
    - This function reads the entire input into memory, which may be inefficient for very large files.
    - The matching is done using string strip() method, which may be sensitive to whitespace differences.
    - If cutting_point1 appears multiple times before cutting_point2, the first occurrence is used.
    """
    lines = code.splitlines(True)
    start, end = -1, -1
    for i, line in enumerate(lines):
        if line.strip() == cutting_point1.strip() and start == -1:
            start = i
        elif line.strip() == cutting_point2.strip():
            end = i
            break
    if start == -1:
        raise ValueError(f"Cutting point '{cutting_point1}' not found in the code.")
    if end == -1:
        raise ValueError(f"Cutting point '{cutting_point2}' not found in the code.")
    return ''.join(lines[:start+1] + [new_code if new_code.endswith('\n') else new_code + '\n'] + lines[end:])

def replace_between(code, cutting_point1, cutting_point2, new_code):
    """
    Replace all code between the lines that match cutting_point1 and cutting_point2 with new_code.

    Args:
    code (str): The original code.
    cutting_point1 (str): The line after which to start replacing.
    cutting_point2 (str): The line before which to stop replacing.
    new_code (str): The code to replace the existing code.

    Returns:
    str: The modified code with the section between cutting_point1 and cutting_point2 replaced.

    Raises:
    ValueError: If either cutting_point1 or cutting_point2 is not found in the code.

    Note:
    - This function reads the entire input into memory, which may be inefficient for very large files.
    - The matching is done using string strip() method, which may be sensitive to whitespace differences.
    - If cutting_point1 appears multiple times before cutting_point2, the first occurrence is used.
    """
    lines = code.splitlines(True)
    start, end = -1, -1
    for i, line in enumerate(lines):
        if line.strip() == cutting_point1.strip() and start == -1:
            start = i
        elif line.strip() == cutting_point2.strip():
            end = i
            break
    if start == -1:
        raise ValueError(f"Cutting point '{cutting_point1}' not found in the code.")
    if end == -1:
        raise ValueError(f"Cutting point '{cutting_point2}' not found in the code.")
    return ''.join(lines[:start+1] + [new_code if new_code.endswith('\n') else new_code + '\n'] + lines[end:])
