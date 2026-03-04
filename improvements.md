# Session Extraction — Minor Improvement Suggestions

These are non-blocking suggestions from the code review of the pickle→JSON session extraction.

## 1. Move `SessionNotFoundError` to `session.py`

`SessionNotFoundError` is conceptually part of the session module but is currently defined in `agents.py`. Moving it to `session.py` and importing it in `agents.py` would improve cohesion.

## 2. Make `generate_session_id` fallback atomic

The fallback path (after 100 collision attempts) generates a longer random ID but skips the `O_CREAT | O_EXCL` atomic check used by the main path. For consistency, the fallback should either use the same atomic pattern or log a warning.

## 3. Document the breaking change from pickle to JSON

Existing users with a `context.pkl` file will find `-r` no longer restores their old session. A changelog entry or migration note should mention this. Optionally, `load_context` could detect and warn about a stale `context.pkl` in the package directory.
