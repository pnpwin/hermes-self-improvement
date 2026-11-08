"""Bounded search for the Heavy mode.

Heavy is a small, explicitly budgeted search over a handful of candidate
revisions, not an evolutionary platform. Its job is to answer one question --
"is any of these bounded revisions better than the incumbent, without a
regression?" -- and to stop as soon as that answer is cheap and stable.

The scoring helpers are pure: callers supply the proposal and evaluation
callables, so every step can be replayed offline from recorded artifacts.
"""
from __future__ import annotations

import random
from typing import Any, Callable

MAX_ROUNDS = 3
MAX_VARIANTS_PER_ROUND = 3
MAX_EVALUATIONS = 8
MAX_CHANGED_CHARS = 8000
STOP_AFTER_FLAT_ROUNDS = 2


class SearchError(ValueError):
    """Raised when a search manifest or a search step breaks the contract."""


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise SearchError("search manifest must be an object")
    allowed = {"rounds", "variants_per_round", "max_evaluations", "max_changed_chars", "seed"}
    unknown = set(manifest) - allowed
    if unknown:
        raise SearchError(f"unknown manifest key(s): {', '.join(sorted(unknown))}")
    resolved = {
        "rounds": _int(manifest.get("rounds", 2), "rounds", 1, MAX_ROUNDS),
        "variants_per_round": _int(
            manifest.get("variants_per_round", 2), "variants_per_round", 1, MAX_VARIANTS_PER_ROUND
        ),
        "max_evaluations": _int(
            manifest.get("max_evaluations", 6), "max_evaluations", 1, MAX_EVALUATIONS
        ),
        "max_changed_chars": _int(
            manifest.get("max_changed_chars", 4000), "max_changed_chars", 1, MAX_CHANGED_CHARS
        ),
        "seed": _int(manifest.get("seed", 0), "seed", 0, 2**31 - 1),
    }
    if resolved["rounds"] * resolved["variants_per_round"] > resolved["max_evaluations"]:
        raise SearchError("rounds x variants_per_round must fit inside max_evaluations")
    return resolved


def _int(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise SearchError(f"{name} must be an integer")
    if not low <= value <= high:
        raise SearchError(f"{name} must be {low}..{high}")
    return value


def score(candidate: dict[str, Any]) -> tuple[int, int, int]:
    """(quality, safety, -cost): higher is better on every component."""
    return (
        _num(candidate, "candidate_passes"),
        -_num(candidate, "critical_regressions"),
        -_num(candidate, "changed_characters"),
    )


def _num(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    if type(value) is bool or not isinstance(value, int):
        raise SearchError(f"{key} must be an integer")
    return value


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep rows no other row dominates on quality, safety and changed text."""
    if not isinstance(rows, list):
        raise SearchError("rows must be a list")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("delta_id"), str):
            raise SearchError("every row needs a delta_id")
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique[row["delta_id"]] = row
    kept: list[dict[str, Any]] = []
    for row in unique.values():
        mine = score(row)
        dominated = False
        for other in unique.values():
            if other is row:
                continue
            theirs = score(other)
            if all(a >= b for a, b in zip(theirs, mine)) and any(
                a > b for a, b in zip(theirs, mine)
            ):
                dominated = True
                break
        if not dominated:
            kept.append(row)
    return sorted(kept, key=lambda row: (score(row), row["delta_id"]), reverse=True)


def select_parents(front: list[dict[str, Any]], *, seed: int, count: int = 2) -> list[dict[str, Any]]:
    """Deterministic selection; the seed is recorded so a replay reproduces it."""
    if count < 1:
        raise SearchError("count must be positive")
    if len(front) <= count:
        return list(front)
    ranked = list(front)
    rng = random.Random(seed)
    top = ranked[0]
    rest = ranked[1:]
    chosen = rng.sample(rest, min(count - 1, len(rest)))
    return [top, *chosen]


def flat_round(rows_this_round: list[dict[str, Any]], incumbent_passes: int) -> bool:
    return all(_num(row, "candidate_passes") <= incumbent_passes for row in rows_this_round)


def run_search(
    *,
    manifest: dict[str, Any],
    incumbent_passes: int,
    base_digest: str,
    propose: Callable[[int, list[dict[str, Any]]], list[dict[str, Any]]],
    evaluate: Callable[[dict[str, Any]], dict[str, Any]],
    budget_left: Callable[[], int],
) -> dict[str, Any]:
    """Run the bounded loop and return every row plus the stopping reason."""
    resolved = validate_manifest(manifest)
    rows: list[dict[str, Any]] = []
    evaluations = 0
    stopped = "rounds_exhausted"
    flat_streak = 0
    for round_index in range(resolved["rounds"]):
        front = pareto_front(rows) if rows else []
        if budget_left() < resolved["variants_per_round"]:
            stopped = "budget_exhausted"
            break
        variants = propose(round_index, select_parents(front, seed=resolved["seed"] + round_index))
        if not isinstance(variants, list) or not variants:
            stopped = "no_variants"
            break
        if len(variants) > resolved["variants_per_round"]:
            raise SearchError("proposer returned more variants than the manifest allows")
        round_rows: list[dict[str, Any]] = []
        for delta in variants:
            if evaluations >= resolved["max_evaluations"]:
                stopped = "evaluation_cap"
                break
            if budget_left() < 1:
                stopped = "budget_exhausted"
                break
            changed = int((delta.get("delta") or {}).get("changed_characters") or 0)
            if changed > resolved["max_changed_chars"]:
                continue
            scored = evaluate(delta)
            evaluations += 1
            row = dict(scored)
            row["delta_id"] = delta["delta_id"]
            row["delta"] = delta.get("delta")
            row["changed_characters"] = changed
            row["base_digest"] = base_digest
            rows.append(row)
            round_rows.append(row)
        if stopped in ("evaluation_cap", "budget_exhausted"):
            break
        if round_rows and flat_round(round_rows, incumbent_passes):
            flat_streak += 1
            if flat_streak >= STOP_AFTER_FLAT_ROUNDS:
                stopped = "no_improvement"
                break
        else:
            flat_streak = 0
    front = pareto_front(rows)
    best = front[0] if front else None
    return {
        "rows": rows,
        "front": front,
        "best": best,
        "stopped_reason": stopped,
        "evaluations": evaluations,
        "manifest": resolved,
        "improved": bool(best and _num(best, "candidate_passes") > incumbent_passes),
    }
