import ast
import black
import textwrap
import re
import io
import tokenize

def syntax_check(code):
    try:
        io_obj = io.StringIO(code)
        list(tokenize.generate_tokens(io_obj.readline))
        ast.parse(code)
        return True
    except (SyntaxError, tokenize.TokenError):
        return False

def format_code(code):
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
        if syntax_check(formatted_code):
            return formatted_code
        else:
            raise SyntaxError("Syntax check failed after formatting")
    except black.parsing.InvalidInput:
        if syntax_check(code):
            return code
        else:
            raise SyntaxError("Syntax check failed on original code")

def read_code(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        raise e

def write_code(file_path, source_code):
    try:
        with open(file_path, "w") as file:
            file.write(source_code)
        return (file_path + " successfully written.")
    except Exception as e:
        raise e

def add_prefix_to_lines(input_string, prefix):
    lines = input_string.splitlines()
    prefixed_lines = [prefix + line for line in lines]
    result_string = '\n'.join(prefixed_lines)
    return result_string

class CodeManipulator(ast.NodeTransformer):
    def __init__(self, target, new_code=None, insert_position=None, action="replace"):
        self.target = target
        self.new_code = new_code
        self.new_code_ast = ast.parse(textwrap.dedent(new_code)).body if new_code else None
        self.insert_position = insert_position
        self.action = action
        self.class_stack = []
        self.found = False

    def visit_Module(self, node):
        if self.target == "" and self.action == "insert" and self.insert_position == "before":
            self.found = True
            return ast.Module(body=self.new_code_ast + node.body, type_ignores=[])
        
        if self.action == "insert" and self.insert_position == "before":
            for i, child in enumerate(node.body):
                if isinstance(child, ast.Import) and any(alias.name == self.target for alias in child.names):
                    self.found = True
                    node.body = node.body[:i] + self.new_code_ast + node.body[i:]
                    return node
        
        self.generic_visit(node)
        if self.new_code_ast and "." not in self.target and self.action == "create" and not any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == self.target for n in node.body):
            node.body.extend(self.new_code_ast)
            self.found = True
        return node

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        current_path = ".".join(self.class_stack)

        if current_path == self.target:
            self.found = True
            if self.action == "replace":
                self.class_stack.pop()
                return self.new_code_ast[0] if self.new_code_ast else None
            elif self.action == "remove":
                self.class_stack.pop()
                return None
            elif self.action == "insert":
                if self.insert_position == "after":
                    node.body.extend(self.new_code_ast)
                elif self.insert_position == "before":
                    node.body = self.new_code_ast + node.body

        if self.action == "create" and current_path == ".".join(self.target.split(".")[:-1]):
            method_name = self.target.split(".")[-1]
            if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method_name for n in node.body):
                node.body.extend(self.new_code_ast)
                self.found = True

        new_body = []
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                result = self.visit_ClassDef(child)
                if result is not None:
                    new_body.append(result)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result = self.visit_FunctionDef(child)
                if result is not None:
                    new_body.extend(result if isinstance(result, list) else [result])
            else:
                new_body.append(self.visit(child))
        node.body = new_body

        self.class_stack.pop()
        return node

    def visit_FunctionDef(self, node):
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node):
        return self._visit_function(node)

    def _visit_function(self, node):
        current_path = ".".join(self.class_stack + [node.name])

        if current_path == self.target:
            self.found = True
            if self.action == "replace":
                return self.new_code_ast[0] if self.new_code_ast else None
            elif self.action == "remove":
                return None
            elif self.action == "insert":
                if self.insert_position == "after":
                    return [node] + self.new_code_ast
                elif self.insert_position == "before":
                    return self.new_code_ast + [node]

        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            current_path = ".".join(self.class_stack + [node.targets[0].id])
            if current_path == self.target:
                self.found = True
                if self.action == "remove":
                    return None
                elif self.action == "replace":
                    return self.new_code_ast[0] if self.new_code_ast else None
                elif self.action == "insert":
                    if self.insert_position == "after":
                        return [node] + self.new_code_ast
                    elif self.insert_position == "before":
                        return self.new_code_ast + [node]
        return node

    def visit_Str(self, node):
        return node

    def visit_Constant(self, node):
        return node

    def visit_Global(self, node):
        if not node.names:
            return None
        return node

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif isinstance(value, list):
                            new_values.extend(value)
                        else:
                            new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

def convert_double_quotes_to_single(code):
    def replace_quotes(match):
        # Check if the match is within an f-string, decorator, print statement, or return statement
        if re.search(r'(f|@\w+\(|print\(|return ).*' + re.escape(match.group(0)) + r'.*[\)]?', code):
            return match.group(0)  # Keep double quotes in f-strings, decorator arguments, print statements, and return statements
        return "'" + match.group(1).replace("'", "\\'") + "'"
    
    # Protect multiline f-strings
    code = re.sub(r'(f""".*?""")', lambda m: m.group(1), code, flags=re.DOTALL)
    code = re.sub(r"(f'''.*?''')", lambda m: m.group(1), code, flags=re.DOTALL)
    
    # Replace double-quoted strings with single-quoted strings, except in f-strings, decorator arguments, print statements, and return statements
    code = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', replace_quotes, code)
    
    return code

def _create_or_replace_code(source_code, address, new_code, create_if_missing, insert_position=None, action="replace"):
    tree = ast.parse(source_code)
    manipulator = CodeManipulator(address, new_code, insert_position, action)
    modified_tree = manipulator.visit(tree)
    if not manipulator.found and not create_if_missing:
        raise ValueError(
            f"The target '{address}' does not exist and 'create_if_missing' is set to False."
        )
    modified_code = ast.unparse(modified_tree)
    formatted_code = format_code(modified_code)
    return convert_double_quotes_to_single(formatted_code)

def create_code(source_code, address, new_code):
    return _create_or_replace_code(source_code, address, new_code, create_if_missing=True, action="create")

def replace_code(source_code, address, new_code):
    return _create_or_replace_code(source_code, address, new_code, create_if_missing=False, action="replace")

def remove_code(source_code, address):
    tree = ast.parse(source_code)
    manipulator = CodeManipulator(address, action="remove")
    modified_tree = manipulator.visit(tree)
    if not manipulator.found:
        raise ValueError(f"The target '{address}' does not exist.")
    modified_code = ast.unparse(modified_tree)
    formatted_code = format_code(modified_code)
    return convert_double_quotes_to_single(formatted_code)

def insert_code_before(source_code, address, new_code):
    return _create_or_replace_code(source_code, address, new_code, create_if_missing=True, insert_position='before', action="insert")

def insert_code_after(source_code, address, new_code):
    return _create_or_replace_code(source_code, address, new_code, create_if_missing=True, insert_position='after', action="insert")

def get_signatures_and_docstrings(source_code):
    """
    Extract all function signatures, including regular functions, async functions, methods,
    and their associated docstrings from the given source code.

    Parameters:
    source_code (str): The source code to analyze.

    Returns:
    str: String representation of the code structure including signatures and docstrings.
    """
    tree = ast.parse(source_code)
    result_lines = []

    class SignatureVisitor(ast.NodeVisitor):
        def visit_Module(self, node):
            module_docstring = ast.get_docstring(node)
            if module_docstring:
                result_lines.append(f"'''{module_docstring}'''\n")
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            self.process_function(node, '')

        def visit_AsyncFunctionDef(self, node):
            self.process_async_function(node, '')

        def visit_ClassDef(self, node, indent=''):
            class_info = {
                'class_name': node.name,
                'docstring': ast.get_docstring(node),
                'methods': []
            }
            result_lines.append(f"{indent}class {node.name}:")
            if class_info['docstring']:
                result_lines.append(f"{indent}    '''{class_info['docstring']}'''\n")

            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    self.process_function(item, indent + '    ')
                elif isinstance(item, ast.AsyncFunctionDef):
                    self.process_async_function(item, indent + '    ')
                elif isinstance(item, ast.ClassDef):
                    self.visit_ClassDef(item, indent + '    ')

        def process_function(self, node, indent):
            signature = f"{indent}def {node.name}({', '.join(arg.arg for arg in node.args.args)}):"
            docstring = ast.get_docstring(node)
            result_lines.append(signature)
            if docstring:
                out_string = f"'''{docstring}'''"
                out_string = add_prefix_to_lines(out_string, f'{indent}    ')
                result_lines.append(out_string)
            else:
                result_lines.append(f'{indent}    pass\n')

        def process_async_function(self, node, indent):
            signature = f"{indent}async def {node.name}({', '.join(arg.arg for arg in node.args.args)}):"
            docstring = ast.get_docstring(node)
            result_lines.append(signature)
            if docstring:
                out_string = f"'''{docstring}'''"
                out_string = add_prefix_to_lines(out_string, f'{indent}    ')
                result_lines.append(out_string)
            else:
                result_lines.append(f'{indent}    pass\n')

    visitor = SignatureVisitor()
    visitor.visit(tree)
    return '\n'.join(result_lines).strip()

def read_code_at_address(source_code, address):
    tree = ast.parse(source_code)
    class ReadCodeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.path = []
            self.code_snippet = None

        def visit_ClassDef(self, node):
            self.path.append(node.name)
            if ".".join(self.path) == address:
                self.code_snippet = ast.get_source_segment(source_code, node)
            self.generic_visit(node)
            self.path.pop()

        def visit_FunctionDef(self, node):
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node):
            self._visit_function(node)

        def _visit_function(self, node):
            self.path.append(node.name)
            if ".".join(self.path) == address:
                start_lineno = min(decorator.lineno for decorator in node.decorator_list) if node.decorator_list else node.lineno
                end_lineno = node.end_lineno
                lines = source_code.splitlines()[start_lineno-1:end_lineno]
                self.code_snippet = "\n".join(lines)
            self.generic_visit(node)
            self.path.pop()

        def visit_Assign(self, node):
            if isinstance(node.targets[0], ast.Name):
                self.path.append(node.targets[0].id)
                if ".".join(self.path) == address:
                    self.code_snippet = ast.get_source_segment(source_code, node)
                self.path.pop()
            self.generic_visit(node)

    visitor = ReadCodeVisitor()
    visitor.visit(tree)
    if visitor.code_snippet:
        return visitor.code_snippet
    else:
        return f"No code found at address '{address}'."

def change_docstring(source_code, address, new_docstring):
    tree = ast.parse(source_code)

    class DocstringChanger(ast.NodeTransformer):
        def __init__(self):
            self.path = []
            self.found = False

        def visit_ClassDef(self, node):
            self.path.append(node.name)
            if ".".join(self.path) == address:
                new_docstring_node = ast.parse(new_docstring).body[0].value
                if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                    node.body[0].value = new_docstring_node
                else:
                    node.body.insert(0, ast.Expr(value=new_docstring_node))
                self.found = True
            self.generic_visit(node)
            self.path.pop()
            return node

        def visit_FunctionDef(self, node):
            return self._visit_function(node)

        def visit_AsyncFunctionDef(self, node):
            return self._visit_function(node)

        def _visit_function(self, node):
            self.path.append(node.name)
            if ".".join(self.path) == address:
                new_docstring_node = ast.parse(new_docstring).body[0].value
                if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                    node.body[0].value = new_docstring_node
                else:
                    node.body.insert(0, ast.Expr(value=new_docstring_node))
                self.found = True
            self.generic_visit(node)
            self.path.pop()
            return node

    changer = DocstringChanger()
    modified_tree = changer.visit(tree)
    if not changer.found:
        raise ValueError(f"The target '{address}' does not exist.")
    modified_code = ast.unparse(modified_tree)
    return format_code(modified_code)
