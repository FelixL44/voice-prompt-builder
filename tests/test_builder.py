"""Tests for the deterministic prompt template."""

from __future__ import annotations

import re

import pytest

from backend.builder import (
    SECTIONS,
    EmptyPromptError,
    build_prompt,
    estimate_tokens,
    used_sections,
)
from backend.schemas import Extraction


@pytest.fixture
def full() -> Extraction:
    return Extraction(
        goal="Draft a first reply to a support ticket",
        audience="Support agents, non-technical",
        context="Zendesk inbox, German and English customers",
        constraints=["Never send automatically", "Handle German and English"],
        examples=["Hallo, danke fuer deine Nachricht"],
        output_format="Reply text plus a confidence score",
        success_criteria=["60% accepted unedited"],
    )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_every_field_appears(full: Extraction) -> None:
    prompt = build_prompt(full)

    assert "Draft a first reply" in prompt
    assert "Support agents" in prompt
    assert "Zendesk inbox" in prompt
    assert "Never send automatically" in prompt
    assert "Hallo, danke" in prompt
    assert "confidence score" in prompt
    assert "60% accepted" in prompt


def test_sections_are_well_formed_xml_blocks(full: Extraction) -> None:
    prompt = build_prompt(full)

    for section in SECTIONS:
        assert f"<{section.tag}>" in prompt
        assert f"</{section.tag}>" in prompt


def test_context_comes_first_and_task_second(full: Extraction) -> None:
    """Long reference material leads; the instruction follows it."""
    prompt = build_prompt(full)

    assert prompt.startswith("<context>")
    assert prompt.index("<task>") < prompt.index("<constraints>")


def test_output_format_sits_near_the_end(full: Extraction) -> None:
    """Formatting instructions land closest to where generation begins."""
    prompt = build_prompt(full)

    assert prompt.index("<output_format>") > prompt.index("<task>")
    assert prompt.index("<output_format>") > prompt.index("<constraints>")


def test_lists_render_as_dashed_items(full: Extraction) -> None:
    prompt = build_prompt(full)

    assert "- Never send automatically" in prompt
    assert "- Handle German and English" in prompt


# ---------------------------------------------------------------------------
# Empty fields
# ---------------------------------------------------------------------------


def test_empty_fields_are_omitted_entirely() -> None:
    """An empty tag tells the model a section exists and then says nothing."""
    prompt = build_prompt(Extraction(goal="Ship it"))

    assert "<task>" in prompt
    assert "<examples>" not in prompt
    assert "<constraints>" not in prompt
    assert "<audience>" not in prompt


def test_whitespace_only_fields_are_omitted() -> None:
    prompt = build_prompt(Extraction(goal="Ship it", audience="   "))

    assert "<audience>" not in prompt


def test_list_of_blanks_is_omitted() -> None:
    prompt = build_prompt(Extraction(goal="Ship it", constraints=["", "  "]))

    assert "<constraints>" not in prompt


def test_a_completely_empty_extraction_is_an_error() -> None:
    with pytest.raises(EmptyPromptError, match="at least one field"):
        build_prompt(Extraction())


def test_used_sections_matches_the_prompt(full: Extraction) -> None:
    partial = Extraction(goal="Ship it", constraints=["Fast"])
    prompt = build_prompt(partial)

    tags = used_sections(partial)
    assert tags == ["task", "constraints"]
    for tag in tags:
        assert f"<{tag}>" in prompt


# ---------------------------------------------------------------------------
# Determinism -- the whole point of not using a model here
# ---------------------------------------------------------------------------


def test_building_twice_gives_identical_output(full: Extraction) -> None:
    assert build_prompt(full) == build_prompt(full)


def test_no_empty_lines_inside_a_section(full: Extraction) -> None:
    """Sections are separated by blank lines; their contents are not."""
    prompt = build_prompt(full)

    for block in re.findall(r"<(\w+)>\n(.*?)\n</\1>", prompt, re.DOTALL):
        assert "\n\n" not in block[1], f"blank line inside <{block[0]}>"


# ---------------------------------------------------------------------------
# Token estimate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected", [("", 0), ("abcd", 1), ("a" * 400, 100), ("abc", 0)]
)
def test_estimate_tokens(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


def test_estimate_scales_with_the_prompt(full: Extraction) -> None:
    prompt = build_prompt(full)

    assert estimate_tokens(prompt) == len(prompt) // 4
    assert estimate_tokens(prompt) > 0
