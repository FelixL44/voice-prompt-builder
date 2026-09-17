"""A single, deliberately rough token estimate, shared across the app.

Real tokenisers are model-specific and importing one would pull in a
dependency this project exists to avoid. Four characters per token is close
enough for the two jobs it has: warning that a prompt is large, and deciding
whether a transcript will fit a context window before sending it.

It is used as a *conservative* bound, never as an exact count.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Approximate the token count of ``text``."""
    return len(text) // CHARS_PER_TOKEN
