"""Exposure and confirmed-outcome telemetry for one skill version.

This is deliberately weak evidence made explicit: a ``skill_view`` is an
exposure, not a success, and an unbound exposure stays ``unknown`` rather than
becoming a zero. Version identity is the package digest, so a summary never
attributes outcomes from one revision to another.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS exposures (
    exposure_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    skill_name TEXT NOT NULL,
    skill_digest TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    observed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    exposure_id TEXT PRIMARY KEY,
    outcome_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    passed INTEGER,
    evidence_ids TEXT NOT NULL,
    bound_at REAL NOT NULL,
    FOREIGN KEY(exposure_id) REFERENCES exposures(exposure_id)
);
CREATE INDEX IF NOT EXISTS exposures_skill ON exposures(skill_name, skill_digest);
"""

OUTCOME_KINDS = ("task_passed", "task_failed", "unknown")


class ExposureError(ValueError):
    """Raised when telemetry would assert more than it can know."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ExposureError(f"{name} must be a non-empty string")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ExposureError("skill_digest must be a sha256 hex digest")
    return value


class ExposureStore:
    def __init__(self, path: str | Path, *, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def record_exposure(
        self,
        *,
        event_id: str,
        skill_name: str,
        skill_digest: str,
        scope_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        _text(event_id, "event_id")
        _text(skill_name, "skill_name")
        _digest(skill_digest)
        _text(scope_id, "scope_id")
        _text(session_id, "session_id")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM exposures WHERE event_id=?", (event_id,)).fetchone()
            if row is not None:
                conn.execute("COMMIT")
                if (row["skill_name"], row["skill_digest"]) != (skill_name, skill_digest):
                    raise ExposureError("the same event_id was recorded for another skill version")
                return {"exposure_id": row["exposure_id"], "duplicate": True}
            exposure_id = f"exposure-{uuid.uuid4().hex[:16]}"
            conn.execute(
                "INSERT INTO exposures(exposure_id, event_id, skill_name, skill_digest, scope_id,"
                " session_id, observed_at) VALUES(?,?,?,?,?,?,?)",
                (exposure_id, event_id, skill_name, skill_digest, scope_id, session_id, self.clock()),
            )
            conn.execute("COMMIT")
            return {"exposure_id": exposure_id, "duplicate": False}
        finally:
            conn.close()

    def bind_confirmed_outcome(
        self,
        exposure_id: str,
        *,
        outcome_id: str,
        kind: str,
        passed: bool | None,
        evidence_ids: Iterable[str],
        expected_digest: str,
    ) -> dict[str, Any]:
        _text(exposure_id, "exposure_id")
        _text(outcome_id, "outcome_id")
        if kind not in OUTCOME_KINDS:
            raise ExposureError(f"unsupported outcome kind: {kind}")
        if kind == "unknown" and passed is not None:
            raise ExposureError("an unknown outcome must not carry a pass/fail value")
        if kind != "unknown" and type(passed) is not bool:
            raise ExposureError("a confirmed outcome needs a boolean passed value")
        ids = [item for item in evidence_ids]
        if not ids or not all(isinstance(item, str) and item for item in ids):
            raise ExposureError("a bound outcome needs at least one evidence id")
        _digest(expected_digest)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM exposures WHERE exposure_id=?", (exposure_id,)).fetchone()
            if row is None:
                conn.execute("COMMIT")
                raise ExposureError("unknown exposure")
            if row["skill_digest"] != expected_digest:
                conn.execute("COMMIT")
                raise ExposureError("skill version changed since the exposure was recorded")
            existing = conn.execute("SELECT * FROM outcomes WHERE exposure_id=?", (exposure_id,)).fetchone()
            if existing is not None:
                conn.execute("COMMIT")
                if existing["outcome_id"] != outcome_id:
                    raise ExposureError("this exposure is already bound to another outcome")
                return {"exposure_id": exposure_id, "bound": True, "duplicate": True}
            conn.execute(
                "INSERT INTO outcomes(exposure_id, outcome_id, kind, passed, evidence_ids, bound_at)"
                " VALUES(?,?,?,?,?,?)",
                (
                    exposure_id,
                    outcome_id,
                    kind,
                    None if passed is None else int(passed),
                    ",".join(ids),
                    self.clock(),
                ),
            )
            conn.execute("COMMIT")
            return {"exposure_id": exposure_id, "bound": True, "duplicate": False}
        finally:
            conn.close()

    def summarize(self, *, skill_name: str) -> dict[str, Any]:
        _text(skill_name, "skill_name")
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT e.skill_digest AS digest, COUNT(*) AS exposures,"
                " SUM(CASE WHEN o.exposure_id IS NULL THEN 1 ELSE 0 END) AS unbound,"
                " SUM(CASE WHEN o.passed = 1 THEN 1 ELSE 0 END) AS passed,"
                " SUM(CASE WHEN o.passed = 0 THEN 1 ELSE 0 END) AS failed"
                " FROM exposures e LEFT JOIN outcomes o ON o.exposure_id = e.exposure_id"
                " WHERE e.skill_name=? GROUP BY e.skill_digest ORDER BY e.skill_digest",
                (skill_name,),
            ).fetchall()
        finally:
            conn.close()
        versions = []
        for row in rows:
            versions.append(
                {
                    "skill_digest": row["digest"],
                    "exposures": row["exposures"],
                    "unbound": row["unbound"],
                    "passed": row["passed"] or 0,
                    "failed": row["failed"] or 0,
                    "unknown": row["unbound"],
                }
            )
        return {
            "skill_name": skill_name,
            "versions": versions,
            "scope": "telemetry only; an exposure is not a success and a version is never merged",
        }


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic offline summary used by the replay gate.

    Duplicate and reordered events must produce the same totals; an exposure
    without a bound outcome stays unknown.
    """
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict):
            raise ExposureError("events must be objects")
        event_id = _text(event.get("event_id"), "event_id")
        kind = event.get("kind")
        if kind not in ("exposure", "outcome"):
            raise ExposureError(f"unsupported event kind: {kind!r}")
        key = (event_id, kind)
        if key in seen:
            continue  # duplicate delivery must not double-count
        seen[key] = event
    totals: dict[str, dict[str, int]] = {}
    bound: dict[str, str] = {}
    exposures: set[str] = set()
    for (event_id, kind), event in seen.items():
        digest = _digest(event.get("skill_digest"))
        bucket = totals.setdefault(digest, {"exposures": 0, "passed": 0, "failed": 0, "unknown": 0})
        if kind == "exposure":
            exposures.add(event_id)
            bucket["exposures"] += 1
            continue
        exposure_event = _text(event.get("exposure_event_id"), "exposure_event_id")
        if exposure_event in bound:
            continue  # first confirmed binding wins; a later duplicate is not a new outcome
        bound[exposure_event] = digest
        if event.get("passed") is True:
            bucket["passed"] += 1
        elif event.get("passed") is False:
            bucket["failed"] += 1
        else:
            bucket["unknown"] += 1
    for digest, bucket in totals.items():
        bound_here = sum(1 for value in bound.values() if value == digest)
        bucket["unknown"] += max(0, bucket["exposures"] - bound_here)
    return {"versions": [{"skill_digest": digest, **totals[digest]} for digest in sorted(totals)]}
