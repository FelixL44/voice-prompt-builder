"""Guard the duplicated logic between the backend and the browser.

The backend decides what is missing when the extraction arrives; the browser
has to make the same judgement live as the user edits. That means the rule
exists twice, so these tests fail the moment the two copies disagree.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from backend.analyze import NULL_STRINGS, is_empty
from backend.config import FRONTEND_DIR
from backend.schemas import Extraction

APP_JS = FRONTEND_DIR / "app.js"

requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is needed to run the frontend logic"
)


def _js_null_strings() -> set[str]:
    """Pull the NULL_STRINGS literal out of app.js."""
    source = APP_JS.read_text()
    block = source.split("const NULL_STRINGS = new Set([")[1].split("]);")[0]
    return set(re.findall(r'"([^"]*)"', block))


def test_null_strings_match_between_backend_and_frontend() -> None:
    """A term the browser does not know would look answered after an edit."""
    assert _js_null_strings() == set(NULL_STRINGS)


def test_frontend_knows_every_extraction_field() -> None:
    """A field missing from FIELDS would be silently uneditable."""
    source = APP_JS.read_text()
    block = source.split("const FIELDS = {")[1].split("\n};")[0]
    declared = set(re.findall(r"^\s*(\w+):\s*\{", block, re.MULTILINE))

    assert declared == set(Extraction.model_fields)


def test_list_fields_are_marked_as_lists() -> None:
    """Editing a list as free text would post a string where a list belongs."""
    source = APP_JS.read_text()
    block = source.split("const FIELDS = {")[1].split("\n};")[0]
    js_lists = {
        name
        for name, flag in re.findall(r"^\s*(\w+):.*list:\s*(true|false)", block, re.MULTILINE)
        if flag == "true"
    }

    expected = {
        name
        for name, field in Extraction.model_fields.items()
        if "list" in str(field.annotation)
    }
    assert js_lists == expected


@requires_node
def test_is_empty_agrees_between_python_and_javascript() -> None:
    """Run the same cases through both implementations and compare."""
    cases: list[object] = [
        None, "", "   ", "null", "NULL", "none.", "N/A", "n/a", "-",
        "unknown", "not specified", "Not Mentioned", [], ["null"], [""],
        "a real answer", ["x"], ["", "x"], "0",
    ]

    script = f"""
    {_extract_js_function()}
    const cases = {json.dumps(cases)};
    console.log(JSON.stringify(cases.map(isEmptyValue)));
    """
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    js_results = json.loads(result.stdout)
    py_results = [is_empty(case) for case in cases]

    mismatches = [
        (case, py, js)
        for case, py, js in zip(cases, py_results, js_results)
        if py != js
    ]
    assert not mismatches, f"python/js disagree on: {mismatches}"


def _referenced_ids(app_js: str) -> set[str]:
    """Element ids app.js looks up, via either the raw DOM call or the helper."""
    return set(re.findall(r'getElementById\("([^"]+)"\)', app_js)) | set(
        re.findall(r'\bel\("([^"]+)"\)', app_js)
    )


def test_every_referenced_element_exists() -> None:
    """Every element app.js looks up must match an id in index.html.

    A typo here yields a null element and a UI control that silently does
    nothing, which is how the model dropdown once shipped empty.
    """
    app_js = APP_JS.read_text()
    html = (FRONTEND_DIR / "index.html").read_text()

    referenced = _referenced_ids(app_js)
    declared = set(re.findall(r'\bid="([^"]+)"', html))

    missing = referenced - declared
    assert not missing, f"app.js references ids not in index.html: {sorted(missing)}"


def test_the_element_check_is_not_vacuous() -> None:
    """Guard the guard.

    Changing how elements are looked up once made the test above match nothing
    and pass regardless, which is worse than not having it.
    """
    referenced = _referenced_ids(APP_JS.read_text())

    assert len(referenced) > 20, (
        f"only {len(referenced)} element references found -- has the lookup "
        "pattern changed? Update _referenced_ids()."
    )


def _extract_js_function() -> str:
    """Lift NULL_STRINGS and isEmptyValue out of app.js so node can run them."""
    source = APP_JS.read_text()
    null_block = "const NULL_STRINGS = new Set([" + \
        source.split("const NULL_STRINGS = new Set([")[1].split("]);")[0] + "]);"
    fn_block = "function isEmptyValue(value) {" + \
        source.split("function isEmptyValue(value) {")[1].split("\n}")[0] + "\n}"
    return f"{null_block}\n{fn_block}"


# ---------------------------------------------------------------------------
# The hidden attribute
# ---------------------------------------------------------------------------


def test_hidden_attribute_is_enforced_in_css() -> None:
    """`hidden` must actually hide, whatever a component rule sets.

    The attribute is applied by the browser's default stylesheet, so any author
    rule setting `display` overrides it. `.status-row { display: flex }` once
    kept the progress row visible from page load, showing placeholder text as
    though a job were running.
    """
    css = (FRONTEND_DIR / "style.css").read_text()

    match = re.search(r"\[hidden\]\s*\{([^}]*)\}", css)
    assert match, "style.css needs a [hidden] rule"

    body = match.group(1).replace(" ", "")
    assert "display:none" in body
    assert "!important" in body, "without !important a component rule still wins"


def test_elements_hidden_by_js_are_declared_hidden_in_html() -> None:
    """An element the JS reveals must start hidden, or it flashes on load."""
    app_js = APP_JS.read_text()
    html = (FRONTEND_DIR / "index.html").read_text()

    revealed = set(re.findall(r'ui\.(\w+)\.hidden\s*=\s*false', app_js))
    id_map = dict(re.findall(r'(\w+):\s*el\("([^"]+)"\)', app_js))

    for name in revealed:
        element_id = id_map.get(name)
        if not element_id:
            continue
        # Find that element's opening tag and check it carries `hidden`.
        tag = re.search(rf'<[^>]*id="{re.escape(element_id)}"[^>]*>', html)
        assert tag, f"no element with id {element_id}"
        # The bare attribute, not aria-hidden, which does not affect layout.
        assert re.search(r'(?<!-)\bhidden\b', tag.group(0)), (
            f"#{element_id} is revealed by JS but does not start hidden"
        )
