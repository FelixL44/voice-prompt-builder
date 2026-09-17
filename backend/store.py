"""Persistent session history, in a local SQLite file.

Replaces the browser-only cache. History that lives in ``localStorage`` is lost
when the cache is cleared and invisible from any other browser, which makes it
useless as a record of work.

SQLite from the standard library, deliberately: no new dependency, one file on
disk, and it stays as local as everything else. A connection is opened per
call rather than shared, because requests run on a threadpool and SQLite
connections are not safe to pass between threads.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    created        REAL NOT NULL,
    updated        REAL NOT NULL,
    title          TEXT NOT NULL DEFAULT '',
    language       TEXT NOT NULL DEFAULT 'en',
    audio_name     TEXT NOT NULL DEFAULT '',
    audio_seconds  REAL NOT NULL DEFAULT 0,
    transcript     TEXT NOT NULL DEFAULT '',
    extraction     TEXT,
    prompt         TEXT NOT NULL DEFAULT '',
    meta           TEXT
);
CREATE INDEX IF NOT EXISTS sessions_updated ON sessions (updated DESC);
"""


class StoreError(Exception):
    """The history database could not be used."""

    code = "store_unavailable"


@contextmanager
def connect(settings: Settings | None = None) -> Iterator[sqlite3.Connection]:
    """Open the history database, creating it and its schema if needed."""
    settings = settings or get_settings()
    path = settings.db_path

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=10)
    except (OSError, sqlite3.Error) as exc:
        raise StoreError(f"Could not open the history database: {exc}") from exc

    conn.row_factory = sqlite3.Row
    try:
        # Requests run on a threadpool, so readers and a writer overlap. WAL
        # lets them proceed together instead of colliding on the default
        # rollback journal and waiting out the busy timeout.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise StoreError(f"History database error: {exc}") from exc
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    # Stored as JSON text; absent and null are both "no extraction yet".
    for key in ("extraction", "meta"):
        raw = data.get(key)
        try:
            data[key] = json.loads(raw) if raw else None
        except ValueError:
            data[key] = None
    return data


def save_session(session: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    """Insert or update a session, returning the stored row.

    Upsert rather than insert: the UI saves the same session repeatedly as it
    moves through transcript, variables and prompt.
    """
    now = time.time()
    session_id = session.get("id") or f"s-{uuid.uuid4().hex[:12]}"

    with connect(settings) as conn:
        existing = conn.execute(
            "SELECT created FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        created = existing["created"] if existing else now

        conn.execute(
            """
            INSERT INTO sessions (id, created, updated, title, language, audio_name,
                                  audio_seconds, transcript, extraction, prompt, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                updated = excluded.updated,
                title = excluded.title,
                language = excluded.language,
                audio_name = excluded.audio_name,
                audio_seconds = excluded.audio_seconds,
                transcript = excluded.transcript,
                extraction = excluded.extraction,
                prompt = excluded.prompt,
                meta = excluded.meta
            """,
            (
                session_id, created, now,
                session.get("title") or "",
                session.get("language") or "en",
                session.get("audio_name") or "",
                float(session.get("audio_seconds") or 0),
                session.get("transcript") or "",
                json.dumps(session["extraction"]) if session.get("extraction") else None,
                session.get("prompt") or "",
                json.dumps(session["meta"]) if session.get("meta") else None,
            ),
        )
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()

    return _row_to_dict(row)


def list_sessions(limit: int = 100, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Summaries, newest first. Transcripts are left out to keep this light."""
    with connect(settings) as conn:
        rows = conn.execute(
            """
            SELECT id, created, updated, title, language, audio_seconds,
                   length(transcript) AS transcript_chars,
                   (prompt != '') AS has_prompt
            FROM sessions ORDER BY updated DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {**dict(row), "has_prompt": bool(row["has_prompt"])}
        for row in rows
    ]


def get_session(session_id: str, settings: Settings | None = None) -> dict[str, Any] | None:
    with connect(settings) as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _row_to_dict(row) if row else None


def delete_session(session_id: str, settings: Settings | None = None) -> bool:
    with connect(settings) as conn:
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return cursor.rowcount > 0


def clear_sessions(settings: Settings | None = None) -> int:
    """Delete every session. Returns how many were removed."""
    with connect(settings) as conn:
        cursor = conn.execute("DELETE FROM sessions")
    return cursor.rowcount


def store_stats(settings: Settings | None = None) -> dict[str, Any]:
    """Session count and database size, for the cache panel."""
    settings = settings or get_settings()
    try:
        with connect(settings) as conn:
            count = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
    except StoreError:
        return {"sessions": 0, "bytes": 0}

    size = settings.db_path.stat().st_size if settings.db_path.exists() else 0
    return {"sessions": count, "bytes": size}
