"""Guard for docs-command discoverability (untracked/report_complete.md, closeout item 2).

The docs tool auto-registers via functions.py getattr dispatch, but the model
can only use what the system prompt advertises — so the base agent prompt
must document the docs subcommands.
"""

from pathlib import Path

YAML_PATH = Path(__file__).resolve().parent.parent / 'agents' / 'basic_agent.yaml'


def _prompt_text():
    return YAML_PATH.read_text()


def test_docs_section_present():
    assert 'Docs Commands' in _prompt_text()


def test_all_docs_subcommands_advertised():
    text = _prompt_text()
    assert 'docs [list]' in text
    assert 'docs search <query>' in text
    assert 'docs tree <source>' in text
    assert 'docs view <source>' in text
