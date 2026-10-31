"""Validate SEARCH/REPLACE blocks against a snapshot before applying edits."""

from dataclasses import dataclass
import re

# A single SEARCH/REPLACE block. Multiple blocks may appear in one command.
_BLOCK_PATTERN = re.compile(
    r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
    re.DOTALL,
)


@dataclass
class ReplacementResult:
    content: str
    applied_blocks: int
    errors: list[str]


def validated_find_replace(source: str, command: str) -> ReplacementResult:
    """Validate all blocks against *source*, then apply valid edits bottom-up.

    Failed searches and every block involved in an overlap are skipped.
    Independent valid blocks still apply. Offsets always refer to the original
    snapshot, so replacements cannot create or remove another block's match.
    """
    blocks = list(_BLOCK_PATTERN.finditer(command))
    if not blocks:
        raise ValueError("Command format is incorrect or missing SEARCH and REPLACE sections.")

    errors = {}
    edits = []
    for i, block in enumerate(blocks, start=1):
        # Preserve the tool's existing whitespace normalization.
        search_text = block.group(1).strip()
        replace_text = block.group(2).strip()
        first_line = search_text.splitlines()[0][:80] if search_text else "(empty)"
        if not search_text:
            errors[i] = f"Block {i}: SEARCH text is empty; include text to replace."
            continue

        start = source.find(search_text)
        if start < 0:
            errors[i] = (
                f"Block {i}: SEARCH text not found in the original content. "
                f"All blocks match the original snapshot; use a separate command "
                f"for dependent edits. First line: {first_line!r}"
            )
            continue

        # Check overlapping occurrences too (str.count only counts disjoint ones).
        count = 1
        next_start = source.find(search_text, start + 1)
        while next_start >= 0:
            count += 1
            next_start = source.find(search_text, next_start + 1)
        if count > 1:
            errors[i] = (
                f"Block {i}: SEARCH text matches {count} locations; "
                f"include more surrounding context to make it unique. "
                f"First line: {first_line!r}"
            )
            continue

        edits.append((start, start + len(search_text), i, replace_text))

    # Reject both sides of every conflict, including conflicts with an edit
    # already rejected by another overlap. Adjacent spans are safe.
    edits.sort()
    conflicts = {}
    for index, (start, end, number, _) in enumerate(edits):
        for other_start, _, other_number, _ in edits[index + 1:]:
            if other_start >= end:
                break
            conflicts.setdefault(number, []).append(other_number)
            conflicts.setdefault(other_number, []).append(number)
    for number, others in conflicts.items():
        errors[number] = (
            f"Block {number}: SEARCH range overlaps block(s) "
            f"{', '.join(map(str, sorted(others)))}; conflicting blocks were skipped."
        )

    result = source
    applied = 0
    for start, end, number, replacement in reversed(edits):
        if number not in errors:
            result = result[:start] + replacement + result[end:]
            applied += 1
    return ReplacementResult(result, applied, [errors[i] for i in sorted(errors)])


def find_replace(source: str, command: str) -> str:
    """Strict string-returning API: raise if any block fails validation.

    Uses the same original-snapshot matching and bottom-to-top application as
    the file tool, but leaves callers such as notes with all-or-nothing updates.
    """
    result = validated_find_replace(source, command)
    if result.errors:
        raise ValueError("\n".join(result.errors))
    return result.content
