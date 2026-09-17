"""Tests for the SQLite session history."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import store
from backend.config import Settings, get_settings
from backend.main import app


@pytest.fixture
def db(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "sessions.db")


@pytest.fixture
def api(tmp_path: Path, monkeypatch) -> TestClient:
    """A client whose history lands in a throwaway database."""
    monkeypatch.setenv("VPB_DB_PATH", str(tmp_path / "api.db"))
    get_settings.cache_clear()
    yield TestClient(app)
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_saving_creates_a_session(db: Settings) -> None:
    saved = store.save_session({"title": "First", "transcript": "hello"}, db)

    assert saved["id"]
    assert saved["title"] == "First"
    assert saved["created"] == saved["updated"]


def test_saving_twice_updates_rather_than_duplicates(db: Settings) -> None:
    """The UI saves the same session after each stage."""
    first = store.save_session({"title": "Draft", "transcript": "hello"}, db)
    store.save_session({**first, "prompt": "<task>x</task>"}, db)

    rows = store.list_sessions(settings=db)
    assert len(rows) == 1
    assert store.get_session(first["id"], db)["prompt"] == "<task>x</task>"


def test_created_is_preserved_across_updates(db: Settings) -> None:
    first = store.save_session({"title": "A"}, db)
    updated = store.save_session({**first, "title": "B"}, db)

    assert updated["created"] == first["created"]
    assert updated["updated"] >= first["updated"]


def test_extraction_round_trips_as_json(db: Settings) -> None:
    extraction = {"goal": "Ship", "constraints": ["fast", "cheap"], "examples": []}

    saved = store.save_session({"extraction": extraction}, db)

    assert store.get_session(saved["id"], db)["extraction"] == extraction


def test_missing_extraction_reads_back_as_none(db: Settings) -> None:
    saved = store.save_session({"title": "no extraction"}, db)

    assert store.get_session(saved["id"], db)["extraction"] is None


def test_listing_is_newest_first(db: Settings) -> None:
    store.save_session({"title": "older"}, db)
    store.save_session({"title": "newer"}, db)

    titles = [row["title"] for row in store.list_sessions(settings=db)]
    assert titles[0] == "newer"


def test_listing_reports_whether_a_prompt_exists(db: Settings) -> None:
    store.save_session({"title": "no prompt"}, db)
    store.save_session({"title": "with prompt", "prompt": "<task>x</task>"}, db)

    by_title = {row["title"]: row for row in store.list_sessions(settings=db)}
    assert by_title["with prompt"]["has_prompt"] is True
    assert by_title["no prompt"]["has_prompt"] is False


def test_deleting_removes_only_that_session(db: Settings) -> None:
    keep = store.save_session({"title": "keep"}, db)
    drop = store.save_session({"title": "drop"}, db)

    assert store.delete_session(drop["id"], db) is True
    assert store.get_session(drop["id"], db) is None
    assert store.get_session(keep["id"], db) is not None


def test_deleting_something_absent_is_false(db: Settings) -> None:
    assert store.delete_session("s-nope", db) is False


def test_clearing_removes_everything(db: Settings) -> None:
    store.save_session({"title": "a"}, db)
    store.save_session({"title": "b"}, db)

    assert store.clear_sessions(db) == 2
    assert store.list_sessions(settings=db) == []


def test_stats_report_count_and_size(db: Settings) -> None:
    store.save_session({"title": "a", "transcript": "x" * 500}, db)

    stats = store.store_stats(db)
    assert stats["sessions"] == 1
    assert stats["bytes"] > 0


def test_the_database_is_created_on_demand(tmp_path: Path) -> None:
    """A fresh install has no data directory."""
    settings = Settings(db_path=tmp_path / "nested" / "deeper" / "s.db")

    store.save_session({"title": "first ever"}, settings)

    assert settings.db_path.exists()


def test_a_corrupt_database_raises_a_typed_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is definitely not sqlite" * 100)

    with pytest.raises(store.StoreError):
        store.list_sessions(settings=Settings(db_path=corrupt))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_session_crud_over_http(api: TestClient) -> None:
    created = api.put("/sessions", json={
        "title": "A session", "transcript": "hello there",
        "extraction": {"goal": "Ship it", "constraints": ["fast"]},
    })
    assert created.status_code == 200
    session_id = created.json()["id"]

    listing = api.get("/sessions").json()
    assert [row["title"] for row in listing] == ["A session"]

    fetched = api.get(f"/sessions/{session_id}").json()
    assert fetched["extraction"]["goal"] == "Ship it"
    assert fetched["extraction"]["constraints"] == ["fast"]

    assert api.delete(f"/sessions/{session_id}").status_code == 200
    assert api.get("/sessions").json() == []


def test_reading_an_unknown_session_is_404(api: TestClient) -> None:
    assert api.get("/sessions/s-nope").status_code == 404


def test_deleting_an_unknown_session_is_404(api: TestClient) -> None:
    assert api.delete("/sessions/s-nope").status_code == 404


def test_clear_all_reports_how_many_went(api: TestClient) -> None:
    api.put("/sessions", json={"title": "one"})
    api.put("/sessions", json={"title": "two"})

    assert api.delete("/sessions").json() == {"deleted": 2}


def test_stats_endpoint(api: TestClient) -> None:
    api.put("/sessions", json={"title": "one"})

    stats = api.get("/sessions/stats").json()
    assert stats["sessions"] == 1
    assert stats["bytes"] > 0


def test_history_survives_a_restart(tmp_path: Path, monkeypatch) -> None:
    """The whole point of moving off localStorage."""
    monkeypatch.setenv("VPB_DB_PATH", str(tmp_path / "persist.db"))
    get_settings.cache_clear()

    try:
        first = TestClient(app)
        first.put("/sessions", json={"title": "before restart", "transcript": "x"})

        second = TestClient(app)   # A new client, same database file.
        assert [r["title"] for r in second.get("/sessions").json()] == ["before restart"]
    finally:
        get_settings.cache_clear()
