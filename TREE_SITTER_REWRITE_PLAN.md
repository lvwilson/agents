# Tree-Sitter Code Manipulator — Implementation Plan (v3)

## Executive Summary

Replace `codemanipulator.py` (Python-only, AST-based, lossy) with a tree-sitter-based
code manipulator that works across multiple languages while preserving comments, formatting,
and whitespace. Consolidate `code_scissors.py` and `findreplace.py` into the same module
as utility functions, creating a single unified code manipulation toolkit.

---

## Problem Statement

The current `codemanipulator.py`:
- **Python-only** — uses `ast` module, cannot handle any other language
- **Lossy** — `ast.unparse()` destroys all comments, formatting, blank lines
- **Fragile quote conversion** — regex-based `convert_double_quotes_to_single` has edge cases
- **Black dependency** — forces opinionated reformatting on every operation

`code_scissors.py` and `findreplace.py` are language-agnostic but have no structural awareness.

## Design Principles

1. **Byte-range surgery** — tree-sitter gives exact byte ranges for every node. All mutations
   are string splicing on the original source bytes. No unparsing, no reformatting. What you
   wrote is what you get back (minus the targeted change).
2. **Language-agnostic core with targeted helpers** — structure is discovered from the tree
   itself (nodes with `name` fields are definitions, nodes with `body` fields are containers).
   A small number of language-specific helpers handle grammars where names are not on the
   `name` field (C++ declarators, Go wrapper nodes, Rust `impl` blocks, JS variable
   declarations).
3. **Preserve existing public API** — `functions.py` callers don't change. Same function
   signatures where possible; removed tools are documented.
4. **Minimal complexity** — no query DSL, no visitor pattern, no AST transformation. Just:
   parse → find node by address → splice bytes.

---

## Architecture

### Single file: `agents/tools/code_manipulator.py`

```
code_manipulator.py
├── Language detection & parser caching
│   ├── _detect_language(file_ext) -> str
│   ├── _get_parser(lang) -> tree_sitter.Parser
│   └── LANGUAGES dict (grammar loader + extensions only)
├── Tree-structure discovery (language-agnostic + targeted helpers)
│   ├── _get_node_name(node) -> str | None
│   ├── _get_scope_children(node) -> list[Node]
│   ├── _unwrap_decorated(node) -> Node
│   ├── _drill_declarator_name(node) -> str | None
│   └── _is_definition(node) -> bool
├── Node resolution
│   └── find_node_by_address(root, address) -> Node | None
├── Core operations (all return modified source string)
│   ├── read_at_address(source, address, lang) -> str
│   ├── replace_at_address(source, address, new_code, lang) -> str
│   ├── remove_at_address(source, address, lang) -> str
│   ├── insert_before_address(source, address, new_code, lang) -> str
│   ├── insert_after_address(source, address, new_code, lang) -> str
│   ├── create_at_address(source, address, new_code, lang) -> str
│   └── get_signatures(source, lang) -> str
├── Utilities (absorbed from code_scissors.py / findreplace.py)
│   ├── find_replace(source, command) -> str
│   ├── insert_before_line(source, line, code) -> str
│   ├── insert_after_line(source, line, code) -> str
│   ├── replace_before_line(source, line, code) -> str
│   ├── replace_after_line(source, line, code) -> str
│   ├── replace_between_lines(source, l1, l2, code) -> str
│   └── insert_between_lines(source, l1, l2, code) -> str
└── File I/O helpers
    ├── read_code(file_path) -> str
    └── write_code(file_path, source) -> str
```

### Language Configuration

Grammar loading and file extensions are language-specific. A small set of
grammar-specific helpers handle naming conventions that differ from the
default `name` field convention.

```python
LANGUAGES = {
    "python": {
        "module": "tree_sitter_python",
        "loader": lambda m: m.language(),
        "extensions": [".py"],
    },
    "javascript": {
        "module": "tree_sitter_javascript",
        "loader": lambda m: m.language(),
        "extensions": [".js", ".jsx", ".mjs"],
    },
    "typescript": {
        "module": "tree_sitter_typescript",
        "loader": lambda m: m.language_typescript(),
        "extensions": [".ts"],
    },
    "tsx": {
        "module": "tree_sitter_typescript",
        "loader": lambda m: m.language_tsx(),
        "extensions": [".tsx"],
    },
    "cpp": {
        "module": "tree_sitter_cpp",
        "loader": lambda m: m.language(),
        "extensions": [".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h"],
    },
    "c_sharp": {
        "module": "tree_sitter_c_sharp",
        "loader": lambda m: m.language(),
        "extensions": [".cs"],
    },
    "go": {
        "module": "tree_sitter_go",
        "loader": lambda m: m.language(),
        "extensions": [".go"],
    },
    "rust": {
        "module": "tree_sitter_rust",
        "loader": lambda m: m.language(),
        "extensions": [".rs"],
    },
}

# Reverse lookup: extension -> language name
_EXT_TO_LANG = {}
for lang, cfg in LANGUAGES.items():
    for ext in cfg["extensions"]:
        _EXT_TO_LANG[ext] = lang
```

New languages are added by installing `tree-sitter-<lang>` and adding a config entry.
Most languages require zero structural code changes — the tree-walking logic is
largely generic. Languages with unusual naming conventions (like C++) may need a
small addition to `_get_node_name`.

---

### Tree-Structure Discovery

The core approach is language-agnostic: use tree-sitter's field conventions
(`name`, `body`, `left`) to discover structure. However, empirical analysis of
the grammars reveals that several languages deviate from the simple `name` field
convention. The discovery logic handles these with a prioritized strategy chain.

#### Empirical Grammar Analysis

| Language | Definition types | Has `name` field? | Has `body` field? | Notes |
|----------|-----------------|-------------------|-------------------|-------|
| **Python** | `function_definition`, `class_definition` | ✓ | ✓ | `decorated_definition` wraps definitions |
| **JavaScript** | `function_declaration`, `class_declaration` | ✓ | ✓ (`body` on class/function) | `export_statement` wraps; `lexical_declaration` has no `name` — child `variable_declarator` does |
| **TypeScript** | Same as JS + type aliases, interfaces | ✓ | ✓ | Same wrapping patterns as JS |
| **C++** | `function_definition`, `class_specifier` | `class_specifier` has `name` ✓ | `class_specifier` has `body` ✓ | `function_definition` has NO `name` — name is on `declarator.declarator` (drilling required). Body field is `body` on `function_definition` ✓. `namespace_definition` has `name` and `body` ✓ |
| **C#** | `class_declaration`, `method_declaration`, `constructor_declaration` | ✓ | ✓ | Works with generic approach. `namespace_declaration` has `name` and `body` ✓ |
| **Go** | `function_declaration`, `method_declaration` | ✓ | ✓ | `type_declaration` wraps `type_spec` (which has `name`). `const_declaration` wraps `const_spec` (which has `name`). `var_declaration` wraps `var_spec` (which has `name`). These wrapper nodes need unwrapping. |
| **Rust** | `function_item`, `struct_item`, `enum_item`, `trait_item` | ✓ | ✓ | `impl_item` has `body` but NO `name` — type is on `type` field. `const_item` has `name` ✓. `static_item` has `name` ✓. |

#### Name Extraction Strategy

```python
def _get_node_name(node) -> str | None:
    """Extract the name of a definition node.

    Strategy chain (tried in order):
    1. node.child_by_field_name("name") — works for most definitions
       across Python, JS, TS, C#, Go, and Rust.
    2. Unwrap decorated/exported/wrapper nodes and recurse.
    3. Drill through declarator chain (C++: function_definition →
       declarator → declarator gives the identifier).
    4. Variable declaration unwrapping (JS/TS: lexical_declaration →
       first variable_declarator → name field).
    5. Go declaration unwrapping (type_declaration → type_spec → name,
       const_declaration → const_spec → name, var_declaration → var_spec → name).
    6. Rust impl_item: use the `type` field as the name (gives the
       implementing type, e.g. "MyStruct" for `impl MyStruct { ... }`).
    7. Assignment left-hand side (Python: expression_statement > assignment).
    8. Return None if the node has no extractable name.
    """
    # Strategy 1: field-based name (covers most definitions)
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")

    # Strategy 2: unwrap decorated/exported wrappers
    inner = _unwrap_decorated(node)
    if inner is not node:
        return _get_node_name(inner)

    # Strategy 3: C++ declarator drilling
    # function_definition → declarator (function_declarator) → declarator (identifier)
    name = _drill_declarator_name(node)
    if name:
        return name

    # Strategy 4: JS/TS variable declarations
    # lexical_declaration / variable_declaration → first variable_declarator → name
    if node.type in ("lexical_declaration", "variable_declaration"):
        for child in node.named_children:
            if child.type == "variable_declarator":
                vname = child.child_by_field_name("name")
                if vname:
                    return vname.text.decode("utf-8")
                break

    # Strategy 5: Go declaration wrappers
    # type_declaration → type_spec → name
    # const_declaration → const_spec → name
    # var_declaration → var_spec → name
    _GO_WRAPPERS = {
        "type_declaration": "type_spec",
        "const_declaration": "const_spec",
        "var_declaration": "var_spec",
    }
    if node.type in _GO_WRAPPERS:
        spec_type = _GO_WRAPPERS[node.type]
        for child in node.named_children:
            if child.type == spec_type:
                spec_name = child.child_by_field_name("name")
                if spec_name:
                    return spec_name.text.decode("utf-8")
                break

    # Strategy 6: Rust impl_item — use the `type` field as name
    # impl MyStruct { ... } → type field is "MyStruct"
    if node.type == "impl_item":
        type_node = node.child_by_field_name("type")
        if type_node:
            return type_node.text.decode("utf-8")

    # Strategy 7: assignment — left-hand side identifier
    left = node.child_by_field_name("left")
    if left and left.type in ("identifier", "name"):
        return left.text.decode("utf-8")

    # Strategy 8: expression_statement wrapping an assignment
    if node.type == "expression_statement" and node.named_child_count == 1:
        return _get_node_name(node.named_children[0])

    return None


def _drill_declarator_name(node) -> str | None:
    """Drill through C/C++ declarator chains to find the name.

    C++ function_definition has no `name` field. Instead:
      function_definition → declarator (function_declarator) →
        declarator (identifier | field_identifier)

    C++ field_declaration (member variables):
      field_declaration → declarator (field_identifier)

    Returns the text of the innermost identifier, or None.
    """
    decl = node.child_by_field_name("declarator")
    if not decl:
        return None

    # Drill down: function_declarator → declarator → ...
    # Stop when we hit an identifier-like node
    seen = set()
    current = decl
    while current and id(current) not in seen:
        seen.add(id(current))
        if current.type in ("identifier", "field_identifier", "type_identifier"):
            return current.text.decode("utf-8")
        # Try the `declarator` field first (function_declarator → declarator)
        inner = current.child_by_field_name("declarator")
        if inner:
            current = inner
            continue
        # Try `name` field (some declarator types have it)
        inner = current.child_by_field_name("name")
        if inner:
            return inner.text.decode("utf-8")
        break

    return None


def _unwrap_decorated(node):
    """If node is a decorator/export wrapper, return the inner definition.

    In Python: decorated_definition -> function_definition | class_definition
    In TS/JS: export_statement -> declaration
    Generic: if a node has exactly one named child that itself has a 'name'
    field, treat it as a wrapper.
    """
    # Python: decorated_definition
    if node.type == "decorated_definition":
        # The definition is the last named child (after decorator nodes)
        for child in reversed(node.named_children):
            if child.child_by_field_name("name"):
                return child
    # JS/TS: export_statement
    if node.type == "export_statement":
        decl = node.child_by_field_name("declaration")
        if decl:
            return decl
    return node


def _get_body_children(node):
    """Get the children of a node's body/block, if it has one.

    Works for: class bodies, function bodies, module/program top-level.
    Returns the node's named children directly if it's a root node,
    or the named children of its 'body' field.

    Special cases:
    - C++ class_specifier: body is field_declaration_list
    - C# class_declaration: body is declaration_list
    - Rust impl_item: body is declaration_list
    - Go: no special handling needed (function bodies are blocks)

    All of these use the `body` field, so the generic approach works.
    """
    # Root nodes (module, program, translation_unit, source_file, compilation_unit)
    if node.parent is None:
        return node.named_children

    # Nodes with a 'body' field (class_definition, function_definition, etc.)
    body = node.child_by_field_name("body")
    if body:
        return body.named_children

    return []


def _is_container(node) -> bool:
    """A node is a container if it has a body field with children, or is root."""
    if node.parent is None:
        return True
    body = node.child_by_field_name("body")
    return body is not None and body.named_child_count > 0
```

#### Language-Specific Naming Patterns Summary

The `_get_node_name` function handles all known patterns through its strategy
chain. Here's what each strategy covers:

| Strategy | Covers | Languages |
|----------|--------|-----------|
| 1. `name` field | Functions, classes, methods, structs, enums, traits, const items | Python, JS, TS, C#, Go (func/method), Rust |
| 2. Unwrap decorated | `decorated_definition`, `export_statement` | Python, JS, TS |
| 3. Declarator drilling | `function_definition`, `field_declaration` | C, C++ |
| 4. Variable declaration | `lexical_declaration`, `variable_declaration` | JS, TS |
| 5. Go wrappers | `type_declaration`, `const_declaration`, `var_declaration` | Go |
| 6. Rust impl type | `impl_item` (uses `type` field as name) | Rust |
| 7. Assignment LHS | `assignment` with identifier LHS | Python |
| 8. Expression statement | `expression_statement` wrapping assignment | Python |

---

### Rust `impl` Block Addressing

Rust's `impl` blocks are a unique structural pattern. An `impl` block is a
container (has `body` with function items) but its "name" is the implementing
type, not a `name` field.

**Addressing scheme:**
- `impl MyStruct { fn new() {} }` → `"MyStruct"` addresses the impl block,
  `"MyStruct.new"` addresses the function inside it.
- When there are multiple `impl` blocks for the same type (e.g. `impl MyStruct`
  and `impl Display for MyStruct`), the first match is returned. This is a
  known limitation — trait impl blocks can be distinguished by using the full
  text `impl Display for MyStruct`, but for address-based operations we use
  the simple type name.
- For trait impls, the `type` field still gives the implementing type. To
  disambiguate, the user can use `read_at_address` to inspect which impl
  block was found, or use `find_replace` for precise targeting.

**Rationale:** This matches the mental model of "MyStruct.new" meaning "the
`new` function associated with MyStruct." The limitation of multiple impl
blocks is acceptable because:
1. Most types have one primary impl block
2. Trait impls are usually small and can be targeted with `find_replace`
3. The LLM can use `get_signatures` to see all impl blocks and choose

---

### Go Method Addressing

Go methods have a receiver parameter but are declared at the top level of the
file, not nested inside a struct definition. This means the tree structure is
flat:

```
source_file
├── type_declaration (MyStruct)
├── method_declaration (GetValue) [receiver: *MyStruct]
├── method_declaration (SetValue) [receiver: *MyStruct]
└── function_declaration (main)
```

**Addressing scheme:**
- `"MyStruct"` → the type declaration
- `"GetValue"` → the method (addressed by name, not by receiver)
- `"main"` → the standalone function

Go methods are NOT addressed as `"MyStruct.GetValue"` because they are not
syntactically nested inside the struct. This matches Go's actual structure
and avoids confusion. The `get_signatures` output will show the receiver
type to help the LLM understand the association.

---

### Core Algorithm: Byte-Range Surgery

Every operation follows the same pattern:

```python
def replace_at_address(source: str, address: str, new_code: str, lang: str = None) -> str:
    source_bytes = source.encode("utf-8")
    tree = _parse(source_bytes, lang)
    node = find_node_by_address(tree.root_node, address)
    if node is None:
        raise ValueError(f"Target '{address}' not found.")
    # Splice: everything before node + new code + everything after node
    return (source_bytes[:node.start_byte] +
            new_code.encode("utf-8") +
            source_bytes[node.end_byte:]).decode("utf-8")
```

No AST unparsing. No reformatting. Comments and whitespace outside the target are untouched.

### Address Resolution Algorithm

```python
def find_node_by_address(root_node, address):
    """Resolve 'ClassName.method_name' to a tree-sitter Node.

    Address format:
    - ""              -> root node (module/program)
    - "func"          -> top-level function named 'func'
    - "Class"         -> top-level class named 'Class'
    - "Class.method"  -> method 'method' inside class 'Class'
    - "Class.Inner.m" -> method 'm' inside nested class 'Inner' inside 'Class'

    Returns the outermost wrapping node (e.g. decorated_definition if present)
    so that byte ranges include decorators.
    """
    if address == "":
        return root_node

    parts = address.split(".")
    current = root_node

    for part in parts:
        children = _get_body_children(current)
        found = None
        for child in children:
            name = _get_node_name(child)
            if name == part:
                found = child
                break
        if found is None:
            return None
        current = found

    return current  # This is the wrapper node (decorated_definition, etc.)
```

**Key invariant:** `_get_node_name` unwraps decorators/exports to find the name,
but the *returned node* from address resolution is always the outermost wrapper.
This means `replace_at_address("Class.method", new_code)` replaces the entire
decorated definition including its decorators — which is the correct behavior
since the caller provides new code that includes decorators.

### Decorated Definition Policy

When a definition has decorators (Python) or is wrapped in an export statement (JS/TS):

- **Name matching:** `_get_node_name` looks through the wrapper to find the name.
- **Byte range:** The wrapper node's byte range is used for all operations.
  This means `read_at_address` returns the decorators too, `replace_at_address`
  replaces decorators too, and `remove_at_address` removes decorators too.
- **Rationale:** This matches the current behavior of `codemanipulator.py`'s
  `read_code_at_address` which computes `start_lineno` from decorator positions.
  It also matches user expectations — when you say "replace method X", you mean
  the whole thing including its `@staticmethod` decorator.

### Empty Address Handling

An empty address (`""`) resolves to the root node (module/program). This supports:

- `insert_before_address(source, "", new_code)` — prepend to file (e.g. add imports)
- `insert_after_address(source, "", new_code)` — append to file
- `read_at_address(source, "")` — read entire file (trivial but consistent)

This replaces the current special case in `CodeManipulator.visit_Module` where
`self.target == ""` is handled separately.

### Assignment Addressability

Module-level and class-level variable assignments are addressable:

```python
# Python:  CONSTANT = 42       -> address "CONSTANT"
# Python:  class Foo:
#              x = 10           -> address "Foo.x"
# JS:      const API_URL = ... -> address "API_URL"
# Go:      const GlobalConst   -> address "GlobalConstant"
# Rust:    const GLOBAL: i32   -> address "GLOBAL_CONSTANT"
# C#:      private int _value  -> address "MyClass._value" (via field_declaration)
```

`_get_node_name` handles this via the strategy chain. The assignment/declaration
node itself is returned, so byte-range surgery replaces the entire statement.

### `create_at_address` Semantics

```python
def create_at_address(source: str, address: str, new_code: str, lang: str = None) -> str:
    """Append new_code to the scope identified by address, if the target doesn't exist.

    Address is split into parent_path + target_name:
    - "Class.method" -> parent="Class", target="method"
    - "func"         -> parent="" (module), target="func"

    Behavior:
    - If target_name already exists in parent scope: NO-OP, return source unchanged.
    - If target_name doesn't exist: append new_code to parent scope's body.
    - If parent scope doesn't exist: raise ValueError.

    For class bodies, new_code is appended after the last child of the body,
    with indentation matching existing siblings.
    """
```

This matches the current `create_code` behavior where existing targets are silently
skipped and missing targets are appended.

### Indentation Handling for `create_at_address`

When appending a new method to a class body, the insertion point is after the last
child of the body. The indentation of the new code must match the class body level.

```python
def _detect_indent(node):
    """Detect the indentation level of a node by looking at its start column."""
    return " " * node.start_point[1]

def _reindent(code, target_indent):
    """Adjust the indentation of code to match target_indent.

    Algorithm:
    1. Find the indentation of the first non-empty line (base_indent).
    2. For each line, replace base_indent prefix with target_indent.
    3. Lines with deeper indentation preserve their relative offset.

    Example:
        code = "def foo():\\n    return 1"  (base_indent = "")
        target_indent = "    "
        result = "    def foo():\\n        return 1"
    """
    lines = code.splitlines(True)
    # Find base indent from first non-empty line
    base_indent = ""
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            base_indent = line[:len(line) - len(stripped)]
            break

    result = []
    for line in lines:
        if not line.strip():
            result.append(line)  # preserve blank lines as-is
        elif line.startswith(base_indent):
            result.append(target_indent + line[len(base_indent):])
        else:
            result.append(target_indent + line.lstrip())
    return "".join(result)
```

For `create_at_address("Class.new_method", code)`:
1. Find the `Class` node
2. Get its body children
3. Detect indentation from the last child's start column
4. Find the insertion byte offset (after last child's end_byte)
5. Ensure a blank line separator, then insert the re-indented new code

The `_reindent` function handles the case where the caller provides code with
any base indentation — it strips the base and applies the target indentation.
This is more robust than requiring the caller to know the exact indentation level.

### Signature Extraction

```python
def get_signatures(source: str, lang: str = None) -> str:
    """Extract function/class signatures and docstrings from source code.

    Returns a string representation matching the existing output format:

        '''Module docstring'''

        class ClassName:
            '''Class docstring'''

            def method_name(self, arg1, arg2):
                '''Method docstring'''

            def other_method(self):
                pass

        def standalone_function(x):
            '''Function docstring'''

    Implementation: recursive tree walk using the same language-agnostic
    helpers (_get_node_name, _get_body_children, _is_container).

    For each definition node:
    1. Extract the first line(s) of the node text up to and including
       the opening delimiter (`:` for Python, `{` for C-family, etc.)
    2. Look for a docstring/doc-comment
    3. Format with proper indentation

    Docstring detection:
    - Python: first child of body that is `expression_statement` containing
      a `string` node (triple-quoted docstring). Output includes the
      triple-quote delimiters to match existing format.
    - JS/TS: JSDoc comment (`/** ... */`) immediately preceding the definition.
      Extracted from the source bytes between the previous sibling's end and
      the definition's start.
    - Rust: `///` doc comments preceding the definition (same byte-range approach).
    - C#: `///` XML doc comments preceding the definition.
    - C++: `///` or `/** ... */` comments preceding the definition.
    - Go: `//` comments immediately preceding the definition (Go convention).
    - If no docstring/doc-comment found, emit 'pass' (Python) or '...' (others)
      as placeholder.
    """
```

The output format is kept compatible with the existing `get_signatures_and_docstrings()`
for Python. For other languages, the same structure is used but signatures reflect
the language's syntax (since we extract the literal text from the source).

### `remove_at_address` — Whitespace Cleanup

When removing a node, we consume exactly one trailing newline to avoid leaving
a blank gap, but avoid being overly aggressive:

```python
def remove_at_address(source: str, address: str, lang: str = None) -> str:
    source_bytes = source.encode("utf-8")
    tree = _parse(source_bytes, lang)
    node = find_node_by_address(tree.root_node, address)
    if node is None:
        raise ValueError(f"Target '{address}' not found.")

    start = node.start_byte
    end = node.end_byte

    # Consume exactly one trailing newline to avoid blank gap
    if end < len(source_bytes) and source_bytes[end:end+1] == b'\n':
        end += 1

    # If the line before start is entirely whitespace (indent of removed node),
    # consume back to the previous newline to remove the blank indent line
    line_start = source_bytes.rfind(b'\n', 0, start)
    if line_start >= 0:
        between = source_bytes[line_start+1:start]
        if not between.strip():
            start = line_start + 1

    return (source_bytes[:start] + source_bytes[end:]).decode("utf-8")
```

This approach:
- Removes exactly one trailing newline (prevents double-blank-line gaps)
- Removes leading whitespace on the same line as the node start (cleans up
  indentation of the removed node)
- Does NOT consume additional blank lines before or after (preserves
  intentional spacing between other definitions)

### Parse Error Policy

When tree-sitter encounters syntax errors, it still produces a partial tree
with `ERROR` nodes. The policy is **best-effort operation**:

```python
def _parse(source_bytes, lang):
    parser = _get_parser(lang)
    tree = parser.parse(source_bytes)
    # We do NOT reject files with parse errors.
    # Tree-sitter produces a partial tree that is usually good enough
    # for address resolution. The LLM may be mid-edit and the file
    # may have temporary syntax errors.
    return tree
```

This is better than the current behavior (which raises `SyntaxError` via
`ast.parse()`) because:
1. The LLM may be making incremental edits to a file with temporary errors
2. Tree-sitter's error recovery is quite good — most of the tree is correct
3. If address resolution fails on a broken tree, the operation raises
   `ValueError("Target not found")` which is a clear, actionable error

---

## Tool Surface Changes

### Removed tool: `replace_docstring_at_address`

The dedicated `change_docstring` / `replace_docstring_at_address` tool is **removed**.

**Rationale:** It was a Python-specific operation (finding the first `ast.Str` node
in a function body). Making it truly language-agnostic is fragile — docstring
conventions differ across languages. Instead, the LLM workflow for changing a
docstring becomes:

1. `read_code_at_address(file, "Class.method")` — see current code including docstring
2. `replace_code_at_address(file, "Class.method", new_code)` — provide the full
   function with the updated docstring

This is one extra step for the LLM but eliminates a language-specific tool and
is more robust (the LLM sees exactly what it's replacing).

### Removed utilities: `syntax_check`, `format_code`, `convert_double_quotes_to_single`

These are artifacts of the AST round-trip approach and are no longer needed.
Syntax checking is implicit — if tree-sitter can parse the file, it's valid
(and we operate best-effort even on partial parses).

### Preserved tools (same signatures)

| `functions.py` function | Calls in new module |
|---|---|
| `read_code_signatures_and_docstrings(file_path)` | `code_manipulator.get_signatures(source, lang)` |
| `read_code_at_address(file_path, address)` | `code_manipulator.read_at_address(source, address, lang)` |
| `replace_code_at_address(file_path, address, new_code)` | `code_manipulator.replace_at_address(source, address, new_code, lang)` |
| `add_code_after_address(file_path, address, new_code)` | `code_manipulator.insert_after_address(source, address, new_code, lang)` |
| `add_code_before_address(file_path, address, new_code)` | `code_manipulator.insert_before_address(source, address, new_code, lang)` |
| `remove_code_at_address(file_path, address)` | `code_manipulator.remove_at_address(source, address, lang)` |
| `find_and_replace(file_path, command)` | `code_manipulator.find_replace(source, command)` |
| `insert_text_after_matching_line(file_path, line, new_code)` | `code_manipulator.insert_after_line(source, line, new_code)` |
| `insert_text_before_matching_line(file_path, line, new_code)` | `code_manipulator.insert_before_line(source, line, new_code)` |
| `replace_text_before_matching_line(file_path, line, new_code)` | `code_manipulator.replace_before_line(source, line, new_code)` |
| `replace_text_after_matching_line(file_path, line, new_code)` | `code_manipulator.replace_after_line(source, line, new_code)` |
| `replace_text_between_matching_lines(file_path, l1, l2, new_code)` | `code_manipulator.replace_between_lines(source, l1, l2, new_code)` |

Language is auto-detected from the file extension in `functions.py`. The `lang`
parameter is passed through so that `code_manipulator` doesn't need to know about
file paths.

---

## What Gets Removed

| File | Disposition |
|------|-------------|
| `codemanipulator.py` | **Deleted** — fully replaced |
| `code_scissors.py` | **Absorbed** — line-based functions moved into new module |
| `findreplace.py` | **Absorbed** — `find_replace()` moved into new module |

## What Stays the Same

| File | Changes |
|------|---------|
| `functions.py` | Update imports to `from . import code_manipulator`. Remove `replace_docstring_at_address`. Add `lang = _detect_lang(file_path)` to address-based calls. |
| `parser.py` | Remove `replace_docstring_at_address` from tool definitions if listed there. |
| All test files | Update imports. Rewrite `assertCodeEqual` to not use black. |

---

## `find_replace` Fix

The current `findreplace.py` strips whitespace from both search and replace text:

```python
search_text = match.group(1).strip()
replace_text = match.group(2).strip()
```

This destroys leading/trailing whitespace, which breaks indentation-sensitive
replacements. The fix:

```python
search_text = match.group(1)
replace_text = match.group(2)

# Remove exactly one leading and one trailing newline (from the marker format)
# but preserve all other whitespace including indentation
if search_text.startswith('\n'):
    search_text = search_text[1:]
if search_text.endswith('\n'):
    search_text = search_text[:-1]
if replace_text.startswith('\n'):
    replace_text = replace_text[1:]
if replace_text.endswith('\n'):
    replace_text = replace_text[:-1]
```

This preserves indentation within the search/replace blocks while removing
the newlines that are artifacts of the marker format (`<<<<<<< SEARCH\n`).

---

## Implementation Steps

### Phase 1: Core Module (estimated: 4-5 hours)

1. **Create `agents/tools/code_manipulator.py`** with:
   - `LANGUAGES` dict (grammar loaders + extensions for all 8 languages)
   - `_detect_language(ext)` and `_get_parser(lang)` with lazy caching
   - Tree helpers: `_get_node_name` (with full 8-strategy chain),
     `_unwrap_decorated`, `_drill_declarator_name`, `_get_body_children`,
     `_is_container`, `_detect_indent`, `_reindent`
   - `find_node_by_address(root, address)` — the central algorithm

2. **Implement core operations** (all byte-range surgery):
   - `read_at_address(source, address, lang=None)` — return node text
   - `replace_at_address(source, address, new_code, lang=None)` — splice
   - `remove_at_address(source, address, lang=None)` — splice + whitespace cleanup
   - `insert_before_address(source, address, new_code, lang=None)` — insert before node
   - `insert_after_address(source, address, new_code, lang=None)` — insert after node
   - `create_at_address(source, address, new_code, lang=None)` — append if missing

3. **Implement `get_signatures(source, lang=None)`** — recursive tree walk with
   format-compatible output and language-aware docstring detection

### Phase 2: Absorb Utilities (estimated: 30 minutes)

4. **Move line-based operations** from `code_scissors.py` into `code_manipulator.py`:
   - `insert_before_line` (was `insert_before`)
   - `insert_after_line` (was `insert_after`)
   - `replace_before_line` (was `replace_before`)
   - `replace_after_line` (was `replace_after`)
   - `insert_between_lines` (was `insert_between`)
   - `replace_between_lines` (was `replace_between`)

5. **Move and fix `find_replace()`** from `findreplace.py` — fix the `.strip()` bug
   to preserve indentation in search/replace blocks

6. **Move `read_code()` and `write_code()`** file I/O helpers (ensure UTF-8)

### Phase 3: Integration (estimated: 1-2 hours)

7. **Update `functions.py`**:
   - Change imports: `from . import code_manipulator`
   - Add language detection helper:
     ```python
     def _detect_lang(file_path):
         ext = os.path.splitext(file_path)[1]
         return code_manipulator.detect_language(ext)
     ```
   - Update all call sites (mechanical renaming)
   - Remove `replace_docstring_at_address` function
   - Pass `lang` to all address-based operations

8. **Update `parser.py`** — remove `replace_docstring_at_address` from tool
   definitions if it's listed there.

### Phase 4: Testing (estimated: 3-4 hours)

9. **Rewrite `tests/test_manipulator.py`**:
   - Remove `format_code` dependency from `setUp` and `assertCodeEqual`
   - Use direct string comparison (normalize leading/trailing whitespace only)
   - Test source code defined with exact formatting (no black round-trip)
   - Add comment preservation test (the key improvement)
   - Add decorated function test (verify decorators included in byte range)
   - Add nested class test
   - Add assignment addressability test
   - Add empty address test

10. **Update `tests/test_find_replace.py`** — change import, add indentation
    preservation test to verify the `.strip()` fix

11. **Update `tests/test_code_scissors.py`** and `tests/test_code_scissors_extended.py`**:
    - Change imports to `from agents.tools.code_manipulator import ...`
    - Rename function references if names changed (e.g. `insert_before` → `insert_before_line`)

12. **Add new language tests**:
    - **JavaScript**: address resolution, replace, remove, signatures, export wrapping,
      variable declaration addressing
    - **TypeScript**: same as JS plus type aliases, interfaces
    - **C++**: namespace/class/function addressing, declarator drilling, member functions,
      signatures
    - **C#**: namespace/class/method addressing, constructor addressing, signatures
    - **Go**: type/function/method addressing, const/var addressing, signatures,
      verify methods are top-level (not nested under struct)
    - **Rust**: function/struct/enum/trait addressing, impl block addressing,
      const/static addressing, signatures
    - **Cross-language**: parameterized test that verifies the same operations
      (read, replace, remove) work across all supported languages
    - **Edge cases**: empty file, file with only comments, single function,
      deeply nested, file with parse errors (best-effort)

13. **Run all tests**, fix any regressions

### Phase 5: Cleanup (estimated: 15 minutes)

14. **Delete** `codemanipulator.py`, `code_scissors.py`, `findreplace.py`
15. **Remove** `black` from runtime dependencies
16. **Keep** `black` as optional test dependency if any existing tests still want it,
    otherwise remove entirely
17. **Update** any documentation or README references

---

## Work Estimate Summary

| Phase | Task | Estimate |
|-------|------|----------|
| 1 | Core module with tree-sitter (8 languages) | 4-5 hours |
| 2 | Absorb line-based utilities + fix find_replace | 30 min |
| 3 | Integration with functions.py | 1-2 hours |
| 4 | Testing (8 languages) | 3-4 hours |
| 5 | Cleanup | 15 min |
| **Total** | | **9-12 hours** |

## Dependencies

Already installed:
- `tree-sitter==0.25.2`
- `tree-sitter-python==0.25.0`
- `tree-sitter-javascript==0.25.0`
- `tree-sitter-typescript==0.23.2`

To install:
- `tree-sitter-cpp==0.23.4`
- `tree-sitter-c-sharp==0.23.1`
- `tree-sitter-go==0.25.0`
- `tree-sitter-rust==0.24.0`

Can be removed after migration:
- `black` (only used by codemanipulator.py for reformatting)

Note: Language grammars are optional — if a grammar package is not installed,
the module raises a clear `ImportError` with installation instructions. Line-based
operations (`find_replace`, `insert_before_line`, etc.) work regardless of
grammar availability.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| **C++ declarator drilling fails for complex declarations** | Medium | The `_drill_declarator_name` function follows the `declarator` field chain until it hits an identifier. This handles `function_declarator`, `pointer_declarator`, `reference_declarator`, etc. Tested against common patterns. Unusual declarations (operator overloads, conversion operators) may not be addressable — acceptable limitation. |
| **Rust multiple impl blocks for same type** | Medium | First match wins. Document as known limitation. Users can use `find_replace` for precise targeting of trait impls. `get_signatures` shows all impl blocks. |
| **Go methods not nested under struct** | Low | Documented as intentional design. Methods addressed by name at top level. `get_signatures` shows receiver type for context. |
| **Decorated definitions have different byte ranges** | High | Explicit policy: always return the outermost wrapper node. `_get_node_name` unwraps for name matching; `find_node_by_address` returns the wrapper. |
| **Indentation of inserted code doesn't match context** | Medium | `create_at_address` uses `_reindent` to detect base indentation of new code and adjust to match sibling indentation. For `insert_before/after`, caller provides correctly indented code (same as current). |
| **Blank line handling around removed/inserted nodes** | Medium | `remove_at_address` consumes exactly one trailing newline and leading whitespace on the same line. Conservative approach avoids collapsing intentional blank lines. |
| **Language grammar not installed** | Low | Graceful error: `raise ImportError(f"Install tree-sitter-{lang}")`. Line-based operations work for any language regardless. |
| **Existing tests expect black-formatted output** | High | Tests rewritten to compare source directly. No black dependency in tests. `assertCodeEqual` uses whitespace-normalized comparison. |
| **`_get_node_name` heuristic fails for unusual node types** | Low | The 8-strategy chain covers all standard patterns across 8 languages. Unusual constructs fall through to `None` and are simply not addressable. |
| **`replace_docstring_at_address` removal breaks workflows** | Low | The LLM can achieve the same result with `read_code_at_address` + `replace_code_at_address`. One extra step but more robust and language-agnostic. |
| **Empty address edge cases** | Low | Explicitly handled: `""` resolves to root node. All operations define behavior for root. |
| **Assignment nodes wrapped in expression_statement** | Medium | `_get_node_name` Strategy 8 handles this: if node is `expression_statement` with one named child, delegate to that child. Verified in Python. |
| **C# namespace nesting** | Low | `namespace_declaration` has `name` and `body` fields, so nested namespaces work naturally: `"MyNamespace.MyClass.MyMethod"`. |
| **C++ namespace nesting** | Low | `namespace_definition` has `name` and `body` fields, same as C#. |
| **find_replace .strip() bug** | High | Fixed during absorption. New implementation preserves indentation. Existing tests updated to verify. |
| **tree-sitter API version mismatch** | Low | Parser creation uses `tree_sitter.Language(capsule)` wrapper. Verified working with tree-sitter 0.25.2. |

---

## Key Design Decisions

### 1. No Reformatting

The current system round-trips through `ast.unparse()` + `black.format_str()`, which
destroys the original source. The new system does **byte-range surgery only**:

- `replace`: splice out old bytes, splice in new bytes
- `remove`: splice out bytes (plus one trailing newline)
- `insert`: splice in new bytes at the right position

This means the tool **never touches code it wasn't asked to touch**. Comments, blank lines,
trailing commas, string quote style — all preserved.

### 2. Language-Agnostic Core with Targeted Helpers

Instead of maintaining per-language lists of "container types", "definition types",
"assignment patterns", etc., we discover structure from the tree using a strategy chain:

- Has a `name` field → it's a definition (most languages)
- Has a `body` field → it's a container
- Has a `declarator` chain → drill to find name (C/C++)
- Is a wrapper node → unwrap to find inner definition (Python decorators, JS exports, Go type/const/var declarations)
- Is a Rust `impl_item` → use `type` field as name
- Has a `left` field with an identifier → it's an assignment (Python)

This means adding a new language typically requires **only a config entry** in `LANGUAGES`.
Languages with unusual naming patterns may need a small addition to the strategy chain,
but the core algorithm remains unchanged.

### 3. Removed `change_docstring` in Favor of `replace_at_address`

A dedicated docstring-replacement tool requires language-specific knowledge of where
docstrings live (Python: first string in body; JS: JSDoc comment before definition;
Rust: `///` comments; C#: `///` XML comments; etc.). Rather than building a fragile
multi-language docstring finder, we rely on the LLM to read the code, modify the
docstring in context, and replace the entire definition.

### 4. File I/O Stays in `functions.py`

The `functions.py` layer handles file reading, writing, error handling, and diff
generation. The `code_manipulator` module operates on strings only (source in,
source out). This separation keeps the core module pure and testable without
filesystem side effects.

`read_code()` and `write_code()` are kept as thin convenience wrappers in
`code_manipulator` for backward compatibility but `functions.py` uses its own
`_read_or_error` / `_write_or_error` helpers.

### 5. Best-Effort on Parse Errors

Unlike the current `ast.parse()` which rejects files with syntax errors,
tree-sitter produces a partial tree. We operate best-effort on partial trees,
which is better for an LLM tool that may be making incremental edits to a
file with temporary syntax errors.

### 6. Go Methods are Top-Level

Go methods are declared at the file level with a receiver parameter, not nested
inside a struct. We respect this structure rather than trying to artificially
group methods under their receiver type. This means `"GetValue"` is the address,
not `"MyStruct.GetValue"`. This matches Go's actual semantics and the tree
structure.

### 7. Rust impl Blocks Use Type Name

Rust `impl` blocks don't have a `name` field — the implementing type is on the
`type` field. We use this as the block's name for addressing purposes. When
multiple impl blocks exist for the same type (common with trait implementations),
the first match wins. This is a pragmatic choice that covers the common case
while documenting the limitation.
