"""Atomic learning-budget ledger (SQLite, runtime-private state).

Two processes competing for the last Lite call must not both be admitted, and a
crash after dispatch must not hand the slot back: the reservation row is written
before the model call and only ever *finished* afterwards.

Counters are per active profile (the ledger lives inside the profile's
self-improvement directory) and never claim to be a provider subscription quota.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    manifest_digest TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    ceiling INTEGER NOT NULL,
    day TEXT NOT NULL,
    created_at REAL NOT NULL,
    closed_at REAL,
    outcome TEXT
);
CREATE TABLE IF NOT EXISTS invocations (
    invocation_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    role TEXT NOT NULL,
    route_json TEXT NOT NULL,
    status TEXT NOT NULL,
    reserved_at REAL NOT NULL,
    finished_at REAL,
    usage_json TEXT
);
CREATE TABLE IF NOT EXISTS day_usage (
    day TEXT NOT NULL,
    bucket TEXT NOT NULL,
    calls INTEGER NOT NULL,
    PRIMARY KEY (day, bucket)
);
CREATE INDEX IF NOT EXISTS invocations_job ON invocations(job_id);
"""

DEFAULT_STALE_SECONDS = 1800

class BudgetError(ValueError):
    """Raised for malformed ledger input, never for a plain denial."""


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BudgetError(f"{name} must be a non-empty string")
    if any(ord(char) < 32 for char in value):
        raise BudgetError(f"{name} contains a control character")
    return value


class Ledger:
    """One SQLite file per profile holding jobs, invocations and day counters."""

    def __init__(self, path: str | Path, *, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        conn.row_factory = sqlite3.Row
        # WAL is a persistent file property set once at open; re-asserting it per
        # connection would take a write lock in the middle of concurrent admission.
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    # -- job admission -------------------------------------------------
    def reserve_job(
        self,
        job_id: str,
        *,
        manifest_digest: str,
        scope_id: str,
        mode: str,
        ceiling: int,
        day: str,
        daily_cap: int,
    ) -> dict[str, Any]:
        _require_text(job_id, "job_id")
        _require_text(manifest_digest, "manifest_digest")
        _require_text(scope_id, "scope_id")
        _require_text(mode, "mode")
        _require_text(day, "day")
        if type(ceiling) is not int or ceiling < 0:
            raise BudgetError("ceiling must be a nonnegative integer")
        if type(daily_cap) is not int or daily_cap < 0:
            raise BudgetError("daily_cap must be a nonnegative integer")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is not None:
                conn.execute("COMMIT")
                if row["manifest_digest"] != manifest_digest:
                    return {
                        "admitted": False,
                        "reason": "job_id_conflict",
                        "job_id": job_id,
                    }
                return {
                    "admitted": True,
                    "idempotent": True,
                    "job_id": job_id,
                    "ceiling": row["ceiling"],
                    "outcome": row["outcome"],
                }
            used = self._used(conn, day, "all")
            if used + ceiling > daily_cap:
                conn.execute("COMMIT")
                return {
                    "admitted": False,
                    "reason": "daily_cap",
                    "used_today": used,
                    "daily_cap": daily_cap,
                    "requested": ceiling,
                }
            conn.execute(
                "INSERT INTO jobs(job_id, manifest_digest, scope_id, mode, ceiling, day, created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (job_id, manifest_digest, scope_id, mode, ceiling, day, self.clock()),
            )
            conn.execute("COMMIT")
            return {
                "admitted": True,
                "idempotent": False,
                "job_id": job_id,
                "ceiling": ceiling,
                "used_today": used,
            }
        finally:
            conn.close()

    # -- per-call reservation ------------------------------------------
    def reserve_call(
        self,
        job_id: str,
        invocation_id: str,
        *,
        role: str,
        route: dict[str, str],
        day: str,
        daily_cap: int,
        lite_daily_cap: int,
        lite: bool = False,
        max_concurrent: int | None = None,
        stale_after_seconds: int = DEFAULT_STALE_SECONDS,
    ) -> dict[str, Any]:
        _require_text(job_id, "job_id")
        _require_text(invocation_id, "invocation_id")
        _require_text(role, "role")
        _require_text(day, "day")
        if not isinstance(route, dict):
            raise BudgetError("route must be an object")
        if max_concurrent is not None and (type(max_concurrent) is not int or max_concurrent < 1):
            raise BudgetError("max_concurrent must be a positive integer")
        if type(stale_after_seconds) is not int or stale_after_seconds < 1:
            raise BudgetError("stale_after_seconds must be a positive integer")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._reap(conn, stale_after_seconds)
            job = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                conn.execute("COMMIT")
                return {"admitted": False, "reason": "unknown_job", "job_id": job_id}
            if job["closed_at"] is not None:
                conn.execute("COMMIT")
                return {"admitted": False, "reason": "job_closed", "job_id": job_id}
            existing = conn.execute(
                "SELECT * FROM invocations WHERE invocation_id=?", (invocation_id,)
            ).fetchone()
            if existing is not None:
                conn.execute("COMMIT")
                if existing["job_id"] != job_id:
                    return {"admitted": False, "reason": "invocation_conflict", "invocation_id": invocation_id}
                # Already reserved: never count a second slot for the same attempt.
                return {
                    "admitted": True,
                    "idempotent": True,
                    "invocation_id": invocation_id,
                    "status": existing["status"],
                }
            if max_concurrent is not None:
                active = conn.execute(
                    "SELECT COUNT(*) AS n FROM invocations WHERE status='reserved'"
                ).fetchone()["n"]
                if active >= max_concurrent:
                    conn.execute("COMMIT")
                    return {
                        "admitted": False,
                        "reason": "concurrency_limit",
                        "active": active,
                        "max_concurrent": max_concurrent,
                    }
            # Every reservation consumes its slot permanently: an ambiguous or
            # timed-out attempt may still have been billed by the provider.
            used_job = conn.execute(
                "SELECT COUNT(*) AS n FROM invocations WHERE job_id=?", (job_id,)
            ).fetchone()["n"]
            if used_job + 1 > job["ceiling"]:
                conn.execute("COMMIT")
                return {
                    "admitted": False,
                    "reason": "job_ceiling",
                    "used_job": used_job,
                    "ceiling": job["ceiling"],
                }
            used_all = self._used(conn, day, "all")
            if used_all + 1 > daily_cap:
                conn.execute("COMMIT")
                return {"admitted": False, "reason": "daily_cap", "used_today": used_all, "daily_cap": daily_cap}
            if lite:
                used_lite = self._used(conn, day, "lite")
                if used_lite + 1 > lite_daily_cap:
                    conn.execute("COMMIT")
                    return {
                        "admitted": False,
                        "reason": "lite_daily_cap",
                        "used_lite_today": used_lite,
                        "lite_daily_cap": lite_daily_cap,
                    }
            conn.execute(
                "INSERT INTO invocations(invocation_id, job_id, role, route_json, status, reserved_at)"
                " VALUES(?,?,?,?,?,?)",
                (invocation_id, job_id, role, json.dumps(route, sort_keys=True), "reserved", self.clock()),
            )
            self._bump(conn, day, "all", 1)
            if lite:
                self._bump(conn, day, "lite", 1)
            conn.execute("COMMIT")
            return {
                "admitted": True,
                "idempotent": False,
                "invocation_id": invocation_id,
                "used_today": used_all + 1,
            }
        finally:
            conn.close()

    def finish_call(
        self,
        invocation_id: str,
        *,
        status: str,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _require_text(invocation_id, "invocation_id")
        if status not in {"completed", "failed", "timeout", "blocked", "ambiguous"}:
            raise BudgetError("Unsupported invocation status")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM invocations WHERE invocation_id=?", (invocation_id,)
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return {"updated": False, "reason": "unknown_invocation"}
            conn.execute(
                "UPDATE invocations SET status=?, finished_at=?, usage_json=? WHERE invocation_id=?",
                (status, self.clock(), json.dumps(usage or {}, sort_keys=True), invocation_id),
            )
            conn.execute("COMMIT")
            return {"updated": True, "invocation_id": invocation_id, "status": status}
        finally:
            conn.close()

    def close_job(self, job_id: str, *, outcome: str) -> dict[str, Any]:
        _require_text(job_id, "job_id")
        _require_text(outcome, "outcome")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return {"closed": False, "reason": "unknown_job"}
            if row["closed_at"] is not None:
                conn.execute("COMMIT")
                return {"closed": True, "idempotent": True, "outcome": row["outcome"]}
            conn.execute(
                "UPDATE jobs SET closed_at=?, outcome=? WHERE job_id=?",
                (self.clock(), outcome, job_id),
            )
            conn.execute("COMMIT")
            return {"closed": True, "idempotent": False, "outcome": outcome}
        finally:
            conn.close()

    # -- read-only views ------------------------------------------------
    def job(self, job_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def job_invocations(self, job_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM invocations WHERE job_id=? ORDER BY reserved_at, invocation_id",
                (job_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def pending_invocations(self, job_id: str) -> list[dict[str, Any]]:
        """Reservations that never reached a terminal status (crash/timed-out client)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM invocations WHERE job_id=? AND status='reserved'"
                " ORDER BY reserved_at, invocation_id",
                (job_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def reap_stale(self, *, older_than_seconds: int = DEFAULT_STALE_SECONDS) -> dict[str, Any]:
        """Mark abandoned reservations ambiguous; the spent budget slot stays spent."""
        if type(older_than_seconds) is not int or older_than_seconds < 1:
            raise BudgetError("older_than_seconds must be a positive integer")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            reaped = self._reap(conn, older_than_seconds)
            conn.execute("COMMIT")
            return {"reaped": reaped, "older_than_seconds": older_than_seconds}
        finally:
            conn.close()

    def status(self, day: str, *, daily_cap: int, lite_daily_cap: int) -> dict[str, Any]:
        conn = self._connect()
        try:
            used_all = self._used(conn, day, "all")
            used_lite = self._used(conn, day, "lite")
        finally:
            conn.close()
        return {
            "day": day,
            "used_model_calls": used_all,
            "remaining_model_calls": max(0, daily_cap - used_all),
            "daily_cap": daily_cap,
            "used_lite_calls": used_lite,
            "remaining_lite_calls": max(0, lite_daily_cap - used_lite),
            "lite_daily_cap": lite_daily_cap,
            "scope": "this profile's local ledger; not a provider subscription quota",
        }

    def _used(self, conn: sqlite3.Connection, day: str, bucket: str) -> int:
        row = conn.execute(
            "SELECT calls FROM day_usage WHERE day=? AND bucket=?", (day, bucket)
        ).fetchone()
        return int(row["calls"]) if row is not None else 0

    def _reap(self, conn: sqlite3.Connection, older_than_seconds: int) -> int:
        """Free the concurrency slot of an abandoned call without refunding it."""
        cutoff = self.clock() - older_than_seconds
        cursor = conn.execute(
            "UPDATE invocations SET status='ambiguous' WHERE status='reserved' AND reserved_at < ?",
            (cutoff,),
        )
        return int(cursor.rowcount or 0)

    def _bump(self, conn: sqlite3.Connection, day: str, bucket: str, delta: int) -> None:
        conn.execute(
            "INSERT INTO day_usage(day, bucket, calls) VALUES(?,?,?)"
            " ON CONFLICT(day, bucket) DO UPDATE SET calls = calls + excluded.calls",
            (day, bucket, delta),
        )
