"""Durable lesson and proposal records for the learning lane.

A lesson is a saved observation plus a causal hypothesis; it is never an
instruction change by itself. ``status`` records that distinction explicitly so
a later Standard cycle can cite the lesson it promoted instead of re-deriving it.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL UNIQUE,
    scope_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    evidence_digest TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    base_digest TEXT NOT NULL,
    candidate_digest TEXT NOT NULL,
    decision TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS lessons_scope ON lessons(scope_id, created_at);
CREATE INDEX IF NOT EXISTS proposals_job ON proposals(job_id);
"""

LESSON_STATUSES = ("observation", "validated")
DECISIONS = ("auto_publish", "human_approval_required", "keep_incumbent")


class StoreError(ValueError):
    """Raised for malformed lesson or proposal records."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StoreError(f"{name} must be a non-empty string")
    return value


def _jsonable(payload: Any, name: str) -> str:
    try:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise StoreError(f"{name} is not JSON serialisable") from exc
    if len(blob) > 200_000:
        raise StoreError(f"{name} exceeds the 200k record limit")
    return blob


class LessonStore:
    def __init__(self, path: str | Path, *, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
            if "evidence_digest" not in columns:  # migration for a ledger created before M5
                conn.execute("ALTER TABLE lessons ADD COLUMN evidence_digest TEXT")
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def save_lesson(
        self,
        *,
        job_id: str,
        scope_id: str,
        lesson: dict[str, Any],
        status: str = "observation",
        evidence_digest: str | None = None,
    ) -> dict[str, Any]:
        _text(job_id, "job_id")
        _text(scope_id, "scope_id")
        if status not in LESSON_STATUSES:
            raise StoreError(f"unsupported lesson status: {status}")
        if not isinstance(lesson, dict):
            raise StoreError("lesson must be an object")
        if evidence_digest is not None and (not isinstance(evidence_digest, str) or len(evidence_digest) != 64):
            raise StoreError("evidence_digest must be a sha256 hex digest")
        blob = _jsonable(lesson, "lesson")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM lessons WHERE job_id=?", (job_id,)).fetchone()
            if row is not None:
                conn.execute("COMMIT")
                if row["payload_json"] != blob or row["status"] != status or row["evidence_digest"] != evidence_digest:
                    raise StoreError("this job already saved a different lesson")
                return {
                    "lesson_id": row["lesson_id"],
                    "job_id": job_id,
                    "status": row["status"],
                    "idempotent": True,
                }
            lesson_id = f"lesson-{uuid.uuid4().hex[:16]}"
            conn.execute(
                "INSERT INTO lessons(lesson_id, job_id, scope_id, status, payload_json, evidence_digest, created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (lesson_id, job_id, scope_id, status, blob, evidence_digest, self.clock()),
            )
            conn.execute("COMMIT")
            return {"lesson_id": lesson_id, "job_id": job_id, "status": status, "idempotent": False}
        finally:
            conn.close()

    def find_by_evidence(self, *, scope_id: str, evidence_digest: str) -> dict[str, Any] | None:
        """An identical evidence set already produced a lesson in this scope."""
        _text(scope_id, "scope_id")
        if not isinstance(evidence_digest, str) or len(evidence_digest) != 64:
            raise StoreError("evidence_digest must be a sha256 hex digest")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM lessons WHERE scope_id=? AND evidence_digest=?"
                " ORDER BY created_at LIMIT 1",
                (scope_id, evidence_digest),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            "lesson_id": row["lesson_id"],
            "job_id": row["job_id"],
            "scope_id": row["scope_id"],
            "status": row["status"],
            "lesson": json.loads(row["payload_json"]),
        }

    def recurring(self, *, scope_id: str | None = None, min_occurrences: int = 2) -> list[dict[str, Any]]:
        """Group lessons by normalized observation; a group is a signal, not a verdict."""
        if type(min_occurrences) is not int or min_occurrences < 2:
            raise StoreError("min_occurrences must be an integer >= 2")
        rows = self.recent_lessons(scope_id=scope_id, limit=200)
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            observation = row["lesson"].get("observation")
            key = normalize_observation(observation)
            if not key:
                continue
            group = groups.setdefault(
                key,
                {"observation": observation, "count": 0, "job_ids": [], "scopes": set(), "confidence": set()},
            )
            group["count"] += 1
            group["job_ids"].append(row["job_id"])
            group["scopes"].add(row["scope_id"])
            group["confidence"].add(row["lesson"].get("confidence"))
        result = []
        for group in groups.values():
            if group["count"] < min_occurrences:
                continue
            result.append(
                {
                    "observation": group["observation"],
                    "count": group["count"],
                    "job_ids": sorted(group["job_ids"]),
                    "scopes": sorted(group["scopes"]),
                    "confidence": sorted(item for item in group["confidence"] if item),
                    "scope": "repeated observation; it still needs a paired cycle before any instruction change",
                }
            )
        return sorted(result, key=lambda item: (-item["count"], item["observation"]))

    def find_lesson(self, *, scope_id: str, observation: str) -> dict[str, Any] | None:
        """Exact normalized-observation lookup inside one scope."""
        _text(scope_id, "scope_id")
        key = normalize_observation(observation)
        if not key:
            return None
        for row in self.recent_lessons(scope_id=scope_id, limit=200):
            if normalize_observation(row["lesson"].get("observation")) == key:
                return row
        return None

    def lesson(self, job_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM lessons WHERE job_id=?", (job_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            "lesson_id": row["lesson_id"],
            "job_id": row["job_id"],
            "scope_id": row["scope_id"],
            "status": row["status"],
            "lesson": json.loads(row["payload_json"]),
            "created_at": row["created_at"],
        }

    def recent_lessons(self, *, scope_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 200:
            raise StoreError("limit must be 1..200")
        conn = self._connect()
        try:
            if scope_id is None:
                rows = conn.execute(
                    "SELECT * FROM lessons ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM lessons WHERE scope_id=? ORDER BY created_at DESC LIMIT ?",
                    (scope_id, limit),
                ).fetchall()
        finally:
            conn.close()
        return [
            {
                "lesson_id": row["lesson_id"],
                "job_id": row["job_id"],
                "scope_id": row["scope_id"],
                "status": row["status"],
                "lesson": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def save_proposal(
        self,
        *,
        job_id: str,
        scope_id: str,
        base_digest: str,
        candidate_digest: str,
        decision: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        _text(job_id, "job_id")
        _text(scope_id, "scope_id")
        _text(base_digest, "base_digest")
        _text(candidate_digest, "candidate_digest")
        if decision not in DECISIONS:
            raise StoreError(f"unsupported decision: {decision}")
        blob = _jsonable(payload, "proposal")
        proposal_id = f"proposal-{uuid.uuid4().hex[:16]}"
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO proposals(proposal_id, job_id, scope_id, base_digest, candidate_digest,"
                " decision, payload_json, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    proposal_id,
                    job_id,
                    scope_id,
                    base_digest,
                    candidate_digest,
                    decision,
                    blob,
                    self.clock(),
                ),
            )
            conn.execute("COMMIT")
        finally:
            conn.close()
        return {"proposal_id": proposal_id, "job_id": job_id, "decision": decision}

    def proposals(self, *, job_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 200:
            raise StoreError("limit must be 1..200")
        conn = self._connect()
        try:
            if job_id is None:
                rows = conn.execute(
                    "SELECT * FROM proposals ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM proposals WHERE job_id=? ORDER BY created_at DESC LIMIT ?",
                    (job_id, limit),
                ).fetchall()
        finally:
            conn.close()
        return [
            {
                "proposal_id": row["proposal_id"],
                "job_id": row["job_id"],
                "base_digest": row["base_digest"],
                "candidate_digest": row["candidate_digest"],
                "decision": row["decision"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]


def normalize_observation(value: Any) -> str:
    """Whitespace- and case-insensitive key for grouping repeated observations."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip().lower()
