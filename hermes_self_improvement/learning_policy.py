"""Canonical learning-mode policy for the Skillcraft learning lane.

The policy is owned by the active Hermes profile: mode availability, budget
ceilings and explicit provider routes for every model stage. It is never
supplied by a tool payload, so a checkbox or a score cannot widen authority.

All functions here are pure and stdlib-only; they never call a model, touch a
skill package or write runtime state.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MODES = ("lite", "standard", "heavy")
EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
ROUTE_ROLES = ("reflector", "curator", "task", "reviewer")
SETTINGS_PLUGIN = "hermes-skillcraft"

DEFAULT_MODE_FLAGS = {"lite_enabled": True, "standard_enabled": True, "heavy_enabled": False}
DEFAULT_BUDGET = {
    "timezone": "Europe/Moscow",
    "daily_model_calls": 24,
    "lite_daily_model_calls": 8,
    "standard_job_model_calls": 14,
    "heavy_job_model_calls": 20,
    "max_concurrent_model_calls": 1,
    "default_job_deadline_seconds": 900,
    # Off by default: the lane may write a package only when the operator opts in.
    "auto_publish_enabled": False,
}
DEFAULT_ROUTES = {role: None for role in ROUTE_ROLES}
# Settings in the same block that belong to the plugin's trial runner, not to this
# policy. The policy ignores them instead of refusing the whole configuration.
OTHER_SETTINGS = frozenset({"primary", "fallback_chain"})

# Job ceilings are ceilings, not grants: a Heavy job also needs a daily
# remainder large enough to hold the whole reservation.
JOB_CEILING_KEYS = {
    "standard": "standard_job_model_calls",
    "heavy": "heavy_job_model_calls",
}


class LearningPolicyError(ValueError):
    """Raised when a policy value cannot be trusted or reached the wire."""


def _home(home: Any = None) -> Path:
    if home is None:
        raw = os.environ.get("HERMES_HOME") or "~/.hermes"
        return Path(raw).expanduser()
    return Path(home).expanduser()


def _coerce_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise LearningPolicyError(f"{name} must be a JSON boolean, not {type(value).__name__}")
    return value


def normalize_mode_flags(raw: Any) -> dict[str, bool]:
    """Return the three canonical booleans, rejecting truthy non-booleans."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise LearningPolicyError("Mode flags must be an object")
    unknown = set(raw) - set(DEFAULT_MODE_FLAGS)
    if unknown:
        raise LearningPolicyError(f"Unknown mode flag(s): {', '.join(sorted(unknown))}")
    flags = dict(DEFAULT_MODE_FLAGS)
    for name in DEFAULT_MODE_FLAGS:
        if name in raw:
            flags[name] = _coerce_bool(raw[name], name)
    return flags


def mode_names(flags: dict[str, bool]) -> tuple[str, ...]:
    return tuple(
        name
        for name in MODES
        if flags.get(f"{name}_enabled") is True
    )


def pipeline_for(flags: dict[str, bool]) -> str:
    """Name the single pipeline for a combination; never two pipelines."""
    names = mode_names(flags)
    if not names:
        return "disabled"
    return "+".join(names)


def _positive_int(value: Any, name: str, *, maximum: int = 10_000) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise LearningPolicyError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise LearningPolicyError(f"{name} must be 1..{maximum}")
    return value


def normalize_budget(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise LearningPolicyError("learning_budget must be an object")
    unknown = set(raw) - set(DEFAULT_BUDGET)
    if unknown:
        raise LearningPolicyError(f"Unknown budget key(s): {', '.join(sorted(unknown))}")
    budget = dict(DEFAULT_BUDGET)
    for key, value in raw.items():
        if key == "timezone":
            budget[key] = _timezone(value)
        elif key == "max_concurrent_model_calls":
            budget[key] = _positive_int(value, key, maximum=8)
        elif key == "default_job_deadline_seconds":
            budget[key] = _positive_int(value, key, maximum=4 * 3600)
        elif key == "auto_publish_enabled":
            if type(value) is not bool:
                raise LearningPolicyError("auto_publish_enabled must be a boolean")
            budget[key] = value
        else:
            budget[key] = _positive_int(value, key, maximum=2000)
    if budget["lite_daily_model_calls"] > budget["daily_model_calls"]:
        raise LearningPolicyError("lite_daily_model_calls cannot exceed daily_model_calls")
    return budget


def _timezone(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise LearningPolicyError("timezone must be a non-empty IANA name")
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise LearningPolicyError(f"Unknown timezone: {value}") from exc
    return value


def route_spec(value: Any, role: str = "route") -> dict[str, str]:
    """Validate one explicit route; secrets and implicit defaults are refused."""
    if not isinstance(value, dict):
        raise LearningPolicyError(f"{role}: route must be an object")
    if set(value) != {"provider", "model", "reasoning"}:
        raise LearningPolicyError(
            f"{role}: route requires exactly provider, model, reasoning; never put secrets here"
        )
    for field in ("provider", "model", "reasoning"):
        text = value[field]
        if not isinstance(text, str) or not text or text != text.strip():
            raise LearningPolicyError(f"{role}: invalid {field}")
        if any(ord(char) < 32 for char in text):
            raise LearningPolicyError(f"{role}: control character in {field}")
    if value["reasoning"] not in EFFORTS:
        raise LearningPolicyError(f"{role}: unsupported reasoning label")
    return {"provider": value["provider"], "model": value["model"], "reasoning": value["reasoning"]}


def normalize_routes(raw: Any) -> dict[str, dict[str, str] | None]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise LearningPolicyError("learning_routes must be an object")
    unknown = set(raw) - set(ROUTE_ROLES)
    if unknown:
        raise LearningPolicyError(f"Unknown route role(s): {', '.join(sorted(unknown))}")
    routes: dict[str, dict[str, str] | None] = dict(DEFAULT_ROUTES)
    for role in ROUTE_ROLES:
        value = raw.get(role)
        routes[role] = None if value is None else route_spec(value, role)
    return routes


def read_plugin_settings(home: Any = None) -> dict[str, Any]:
    """Read this plugin's settings leaves from the active profile config."""
    path = _home(home) / "config.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml
    except Exception:  # pragma: no cover - PyYAML is present in the Hermes runtime
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    entries = ((data.get("plugins") or {}).get("entries") or {})
    entry = entries.get(SETTINGS_PLUGIN) or {}
    settings = entry.get("settings") if isinstance(entry, dict) else None
    return settings if isinstance(settings, dict) else {}


def settings_problem(home: Any = None) -> str | None:
    """Why the profile settings could not be read, or ``None`` when they were.

    ``read_plugin_settings`` fails closed to defaults, which is safe but silent:
    a hand-edited ``config.yaml`` with a tab or a duplicate key would quietly run
    the lane on defaults while the operator believes their settings apply.
    """
    path = _home(home) / "config.yaml"
    if not path.is_file():
        return None
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - PyYAML ships with the host
        return f"yaml is unavailable, so settings were ignored: {exc}"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return f"config.yaml is not valid YAML, so settings were ignored: {exc}"
    entries = ((data.get("plugins") or {}).get("entries") or {}) if isinstance(data, dict) else {}
    entry = entries.get(SETTINGS_PLUGIN)
    if entry is None:
        return None
    if not isinstance(entry, dict):
        return f"plugins.entries.{SETTINGS_PLUGIN} is not a mapping, so settings were ignored"
    settings = entry.get("settings")
    if settings is not None and not isinstance(settings, dict):
        return f"plugins.entries.{SETTINGS_PLUGIN}.settings is not a mapping, so settings were ignored"
    return None


def build_policy(
    settings: dict[str, Any] | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical policy from profile settings plus explicit overrides."""
    base = dict(settings or {})
    allowed = set(DEFAULT_MODE_FLAGS) | {"learning_budget", "learning_routes"}
    if overrides:
        unknown = set(overrides) - allowed
        if unknown:
            raise LearningPolicyError(f"Unknown policy override(s): {', '.join(sorted(unknown))}")
        base.update(overrides)
    # ``settings`` is one shared block: the trial runner owns these keys and the
    # policy neither reads nor interprets them. Ignoring them keeps a documented
    # configuration from breaking the lane; every other key stays refused.
    unknown_keys = set(base) - allowed - OTHER_SETTINGS
    if unknown_keys:
        raise LearningPolicyError(f"Unknown policy key(s): {', '.join(sorted(unknown_keys))}")
    flags: dict[str, Any] = {}
    for name in DEFAULT_MODE_FLAGS:
        if name in base:
            flags[name] = base[name]
    policy = {
        "modes": normalize_mode_flags(flags),
        "budget": normalize_budget(base.get("learning_budget")),
        "routes": normalize_routes(base.get("learning_routes")),
    }
    policy["pipeline"] = pipeline_for(policy["modes"])
    policy["enabled_modes"] = mode_names(policy["modes"])
    return policy


def load_policy(home: Any = None, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_policy(read_plugin_settings(home), overrides=overrides)


def clone(policy: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(policy)


def job_role_sequence(pipeline: str) -> tuple[str, ...]:
    """Model roles a pipeline may use, in order; each still needs a route."""
    if pipeline == "disabled":
        return ()
    roles: list[str] = ["reflector"]
    if "standard" in pipeline or "heavy" in pipeline:
        roles.append("curator")
    if "standard" in pipeline:
        roles.append("task")
    if "heavy" in pipeline:
        roles.extend(["task", "reviewer"])
    seen: list[str] = []
    for role in roles:
        if role not in seen:
            seen.append(role)
    return tuple(seen)


def require_routes(policy: dict[str, Any], pipeline: str, *, dry_run: bool = False) -> dict[str, Any]:
    """Fail closed when an enabled stage has no explicit route."""
    if pipeline == "disabled":
        return {"ok": False, "state": "learning_disabled", "missing": []}
    missing = [
        role
        for role in job_role_sequence(pipeline)
        if policy["routes"].get(role) is None
    ]
    if missing:
        return {"ok": False, "state": "route_unconfigured", "missing": missing, "dry_run": dry_run}
    return {"ok": True, "state": "ready", "missing": [], "dry_run": dry_run}


def day_key(policy: dict[str, Any], when) -> str:
    """Local calendar day in the configured zone; never the process locale."""
    zone = ZoneInfo(policy["budget"]["timezone"])
    return when.astimezone(zone).date().isoformat()


def job_ceiling_key(pipeline: str) -> str:
    if "heavy" in pipeline:
        return JOB_CEILING_KEYS["heavy"]
    return JOB_CEILING_KEYS["standard"]


def lite_ceiling(policy: dict[str, Any]) -> int:
    """Lite is one extraction invocation per cycle."""
    return 1


def summarize(policy: dict[str, Any]) -> dict[str, Any]:
    """Compact, secret-free view for status output."""
    return {
        "modes": dict(policy["modes"]),
        "pipeline": policy["pipeline"],
        "timezone": policy["budget"]["timezone"],
        "daily_model_calls": policy["budget"]["daily_model_calls"],
        "lite_daily_model_calls": policy["budget"]["lite_daily_model_calls"],
        "routes_configured": sorted(role for role, route in policy["routes"].items() if route),
        "routes_missing": sorted(role for role, route in policy["routes"].items() if not route),
    }
