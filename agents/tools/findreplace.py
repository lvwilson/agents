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
    * Blocks are applied **sequentially**, in command order: each
      block's search text is looked up in the result of applying all
      preceding blocks.  This lets a later block match text introduced
      by an earlier block's replacement — the natural way to express a
      chain of dependent edits.
    * Each block's search text must occur exactly once in the running
      text, otherwise a ``ValueError`` is raised and **no** replacement
      is returned (all-or-nothing: the caller's original string is
      never mutated, so a raised error leaves the file untouched).
    """
    blocks = list(_BLOCK_PATTERN.finditer(command))
    if not blocks:
        raise ValueError("Command format is incorrect or missing SEARCH and REPLACE sections.")

    result = source
    for i, block in enumerate(blocks, start=1):
        search_text = block.group(1).strip()
        replace_text = block.group(2).strip()

        count = result.count(search_text)
        if count == 0:
            raise ValueError(
                f"Block {i}: SEARCH text not found. "
                f"(Blocks apply sequentially — the text may have been "
                f"altered by an earlier block, or indentation/exact "
                f"content may be wrong.) First line: "
                f"{search_text.splitlines()[0][:80] if search_text else '(empty)'!r}"
            )
        if count > 1:
            raise ValueError(
                f"Block {i}: SEARCH text matches {count} locations; "
                f"include more surrounding context to make it unique. First line: "
                f"{search_text.splitlines()[0][:80]!r}"
            )

        start = result.index(search_text)
        result = result[:start] + replace_text + result[start + len(search_text):]
    return result
