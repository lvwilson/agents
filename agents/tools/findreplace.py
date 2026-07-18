import re

# A single SEARCH/REPLACE block. Multiple blocks may appear in one command.
_BLOCK_PATTERN = re.compile(
    r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
    re.DOTALL,
)


def find_replace(source: str, command: str) -> str:
    """Apply every SEARCH/REPLACE block in *command* to *source*.

    Semantics
    ---------
    * All blocks are extracted with ``re.finditer`` (previously only the
      first block was applied — a silent data-loss bug).
    * Each block's search text must occur in the *original* source, and
      at most once, otherwise a ``ValueError`` is raised and **no**
      replacement is performed (all-or-nothing).  A search text that
      silently matches nothing used to pass as a "successful" no-op,
      hiding agent mistakes.
    * Replacements are applied right-to-left by match position so that
      earlier matches keep their offsets and overlapping blocks are
      handled deterministically.
    """
    blocks = list(_BLOCK_PATTERN.finditer(command))
    if not blocks:
        raise ValueError("Command format is incorrect or missing SEARCH and REPLACE sections.")

    # Collect (start, end, replacement) edits against the original source.
    edits = []
    for i, block in enumerate(blocks, start=1):
        search_text = block.group(1).strip()
        replace_text = block.group(2).strip()

        count = source.count(search_text)
        if count == 0:
            raise ValueError(
                f"Block {i}: SEARCH text not found in source. "
                f"Check indentation/exact content. First line: "
                f"{search_text.splitlines()[0][:80] if search_text else '(empty)'!r}"
            )
        if count > 1:
            raise ValueError(
                f"Block {i}: SEARCH text matches {count} locations; "
                f"include more surrounding context to make it unique. First line: "
                f"{search_text.splitlines()[0][:80]!r}"
            )

        start = source.index(search_text)
        edits.append((start, start + len(search_text), replace_text))

    # Apply right-to-left so earlier positions stay valid.
    result = source
    for start, end, replace_text in sorted(edits, key=lambda e: e[0], reverse=True):
        result = result[:start] + replace_text + result[end:]
    return result
