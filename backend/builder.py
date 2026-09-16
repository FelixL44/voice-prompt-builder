"""Assemble the final prompt from an extraction.

Deliberately not an LLM. The model's job ended once the transcript was sorted
into fields; turning those fields into a prompt is pure string assembly, so it
is reproducible, instant, diffable, and testable. Running it twice on the same
extraction gives byte-identical output.

Section order follows the one structural rule that reliably matters for large
models: long reference material goes first, and the instruction that says what
to *do* comes after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from backend.schemas import Extraction

# A rough proxy used everywhere in this project. Real tokenisers are
# model-specific; four characters per token is close enough to warn someone
# that a prompt is large, which is all this number is for.
CHARS_PER_TOKEN = 4


class EmptyPromptError(Exception):
    """The extraction holds nothing to build a prompt from."""

    code = "empty_prompt"


@dataclass(frozen=True)
class Section:
    """One XML-style block in the assembled prompt."""

    tag: str
    field: str
    render: Callable[[object], str]


def _render_text(value: object) -> str:
    """A prose field: emitted as-is, just tidied."""
    return str(value).strip()


def _render_list(value: object) -> str:
    """A list field: one dash-prefixed item per line."""
    items = value if isinstance(value, (list, tuple)) else [value]
    return "\n".join(f"- {str(item).strip()}" for item in items if str(item).strip())


# Order matters and is fixed.
#
# 1. context first -- it is the longest block and the one the model should read
#    as reference rather than as instruction.
# 2. task next: after the background, before the qualifiers.
# 3. audience, constraints and examples shape *how* the task is done.
# 4. output_format and success_criteria come last, closest to generation, which
#    is where a model is most likely to still be honouring them.
SECTIONS: tuple[Section, ...] = (
    Section("context", "context", _render_text),
    Section("task", "goal", _render_text),
    Section("audience", "audience", _render_text),
    Section("constraints", "constraints", _render_list),
    Section("examples", "examples", _render_list),
    Section("output_format", "output_format", _render_text),
    Section("success_criteria", "success_criteria", _render_list),
)


def _is_blank(value: object) -> bool:
    """True for values that would produce an empty section."""
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return not any(str(item).strip() for item in value)
    return not str(value).strip()


def build_prompt(extraction: Extraction) -> str:
    """Assemble the prompt. Empty fields are omitted rather than left blank.

    An empty ``<examples />`` tag is worse than no tag at all: it tells the
    model a section exists and then says nothing, which invites it to fill the
    gap itself.

    Raises:
        EmptyPromptError: nothing in the extraction has any content.
    """
    data = extraction.model_dump()
    blocks: list[str] = []

    for section in SECTIONS:
        value = data.get(section.field)
        if _is_blank(value):
            continue
        body = section.render(value)
        if not body:
            continue
        blocks.append(f"<{section.tag}>\n{body}\n</{section.tag}>")

    if not blocks:
        raise EmptyPromptError(
            "There is nothing to build a prompt from yet. "
            "Fill in at least one field above."
        )

    return "\n\n".join(blocks)


def estimate_tokens(text: str) -> int:
    """A rough token count: characters divided by four."""
    return len(text) // CHARS_PER_TOKEN


def used_sections(extraction: Extraction) -> list[str]:
    """Tags that the prompt will actually contain, in order."""
    data = extraction.model_dump()
    return [s.tag for s in SECTIONS if not _is_blank(data.get(s.field))]
