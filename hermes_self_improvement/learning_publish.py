"""Promotion gate and publish hand-off for a candidate skill revision.

Two separate decisions live here:

* ``assess_candidate`` is pure. It decides whether an automatically generated
  revision is even eligible for unattended publication.
* ``publish_candidate`` performs the write, and only through an injected native
  writer. When no native writer is available the lane stops and asks for a human
  decision instead of reaching around the host's own tooling.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

DECISIONS = ("auto_publish", "human_approval_required", "keep_incumbent")
SCAN_STATUSES = ("pass", "caution", "fail")


class PublishError(ValueError):
    """Raised when a publish is attempted outside the gate's contract."""


class PublishDeferred(RuntimeError):
    """The writer declined or staged the write; a human must decide.

    Distinct from :class:`PublishError`: nothing is wrong with the candidate or
    the gate, so the cycle is not an invalid stage. It simply did not write.
    """


def _count(value: Any, name: str) -> int:
    if type(value) is bool or not isinstance(value, int) or value < 0:
        raise PublishError(f"{name} must be a nonnegative integer")
    return value


def assess_candidate(
    *,
    base_digest: str,
    candidate_digest: str,
    comparison: dict[str, Any],
    holdout: dict[str, Any],
    cases: dict[str, Any],
    scan: dict[str, Any],
) -> dict[str, Any]:
    """Return a decision plus the exact gate reasons behind it."""
    for name, value in (("base_digest", base_digest), ("candidate_digest", candidate_digest)):
        if not isinstance(value, str) or len(value) != 64:
            raise PublishError(f"{name} must be a sha256 hex digest")
    if not isinstance(comparison, dict) or not isinstance(holdout, dict):
        raise PublishError("comparison and holdout must be objects")
    if not isinstance(cases, dict) or not isinstance(scan, dict):
        raise PublishError("cases and scan must be objects")

    installed = _count(comparison.get("installed_passes"), "installed_passes")
    candidate = _count(comparison.get("candidate_passes"), "candidate_passes")
    regressions = _count(comparison.get("critical_regressions"), "critical_regressions")
    pairs = _count(comparison.get("pairs"), "pairs")
    eligible = comparison.get("eligible")
    if type(eligible) is not bool:
        raise PublishError("comparison.eligible must be a boolean")

    holdout_pairs = _count(holdout.get("pairs", 0), "holdout.pairs")
    holdout_installed = _count(holdout.get("installed_passes", 0), "holdout.installed_passes")
    holdout_candidate = _count(holdout.get("candidate_passes", 0), "holdout.candidate_passes")
    untouched = holdout.get("untouched")
    if type(untouched) is not bool:
        raise PublishError("holdout.untouched must be a boolean")

    trusted = _count(cases.get("trusted", 0), "cases.trusted")
    total = _count(cases.get("total", 0), "cases.total")
    status = scan.get("status")
    if status not in SCAN_STATUSES:
        raise PublishError("scan.status must be pass, caution or fail")

    gates = {
        "digest_bound": candidate_digest != base_digest,
        "paired_comparison_eligible": eligible and pairs > 0,
        "no_critical_regression": regressions == 0,
        "trusted_cases_present": trusted > 0 and total >= trusted,
        "holdout_untouched": untouched and holdout_pairs > 0,
        "scanner_clean": status == "pass",
        "quality_gain": candidate > installed,
        "holdout_not_worse": holdout_candidate >= holdout_installed,
    }
    reasons = sorted(name for name, ok in gates.items() if not ok)
    evidence = {
        "installed_passes": installed,
        "candidate_passes": candidate,
        "pairs": pairs,
        "critical_regressions": regressions,
        "holdout_pairs": holdout_pairs,
        "holdout_installed_passes": holdout_installed,
        "holdout_candidate_passes": holdout_candidate,
        "trusted_cases": trusted,
        "total_cases": total,
        "scanner_status": status,
    }

    if candidate_digest == base_digest:
        return {"decision": "keep_incumbent", "reasons": ["no_change"], "gates": gates, "evidence": evidence}
    if regressions > 0:
        return {
            "decision": "keep_incumbent",
            "reasons": ["no_critical_regression"],
            "gates": gates,
            "evidence": evidence,
        }
    if not gates["quality_gain"]:
        return {
            "decision": "keep_incumbent",
            "reasons": ["quality_gain"],
            "gates": gates,
            "evidence": evidence,
        }
    if reasons:
        return {
            "decision": "human_approval_required",
            "reasons": reasons,
            "gates": gates,
            "evidence": evidence,
        }
    return {"decision": "auto_publish", "reasons": [], "gates": gates, "evidence": evidence}


def publish_candidate(
    *,
    decision: dict[str, Any],
    delta: dict[str, Any],
    staging_root: Any,
    target_root: Any,
    native_writer: Callable[..., dict[str, Any]] | None,
    expected_base_digest: str,
    expected_candidate_digest: str,
) -> dict[str, Any]:
    """Publish only an ``auto_publish`` decision, only through a native writer."""
    from . import skill_delta as sd

    if not isinstance(decision, dict) or decision.get("decision") != "auto_publish":
        raise PublishError("publishing requires an auto_publish decision")
    if not callable(native_writer):
        raise PublishError("native_writer_unavailable: refusing to write without the host's skill tool")
    staging = Path(staging_root).resolve()
    target = Path(target_root).resolve()
    staging_digest = sd.package_digest(staging)
    target_digest = sd.package_digest(target)
    if staging_digest != expected_candidate_digest:
        raise PublishError("staging revision no longer matches the assessed candidate")
    if target_digest != expected_base_digest:
        raise PublishError("installed skill drifted since the candidate was produced")

    result = native_writer(delta=delta, target_root=str(target), base_digest=target_digest)
    if not isinstance(result, dict) or result.get("published") is not True:
        reason = result.get("reason") if isinstance(result, dict) else None
        raise PublishDeferred(f"native writer did not publish: {reason or 'unconfirmed'}")

    after = sd.package_digest(target)
    if after != expected_candidate_digest:
        raise PublishError("published revision does not match the assessed candidate")
    return {
        "published": True,
        "base_digest": target_digest,
        "candidate_digest": after,
        "verification": "digest_match",
        "writer": result.get("writer", "native"),
    }


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
