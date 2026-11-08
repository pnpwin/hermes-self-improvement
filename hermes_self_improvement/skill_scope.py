"""Project-vs-home scope resolution for the learning lane.

Every learning job is bound to exactly one scope. The scope decides where a
generated skill may be written, so it is resolved once, recorded in the job
manifest and never inferred again from free text.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

KIND_PROJECT = "project"
KIND_HOME = "home"
KIND_NONE = "none"
KINDS = (KIND_PROJECT, KIND_HOME, KIND_NONE)

_HOME_SKILLS = (".agents", "skills")
MAX_DEPTH = 64


class ScopeError(ValueError):
    """Raised when a scope cannot be resolved unambiguously."""


def _home(home: Any = None) -> Path:
    if home is None:
        raw = os.environ.get("HERMES_HOME") or "~/.hermes"
        return Path(raw).expanduser().resolve()
    return Path(home).expanduser().resolve()


def _is_git_root(path: Path) -> bool:
    marker = path / ".git"
    return marker.is_dir() or marker.is_file()


def resolve_scope(cwd: Any, home: Any = None) -> dict[str, Any]:
    """Resolve the nearest git project, never the agent home itself."""
    if cwd is None:
        raise ScopeError("cwd is required: a project skill is written only inside an explicit project")
    root_path = Path(cwd).expanduser().resolve()
    home_path = _home(home)
    if not root_path.is_dir():
        raise ScopeError(f"cwd is not a directory: {root_path}")
    if root_path == home_path:
        return {
            "kind": KIND_HOME,
            "root": str(home_path),
            "skills_root": str(home_path.joinpath(*_HOME_SKILLS)),
            "project_slug": None,
        }
    current = root_path
    for _ in range(MAX_DEPTH):
        if current == home_path:
            break
        if _is_git_root(current):
            slug = current.name.strip() or "project"
            return {
                "kind": KIND_PROJECT,
                "root": str(current),
                "skills_root": str(current / ".agents" / "skills"),
                "project_slug": _slug(slug),
            }
        parent = current.parent
        if parent == current:
            break
        current = parent
    return {
        "kind": KIND_HOME,
        "root": str(home_path),
        "skills_root": str(home_path.joinpath(*_HOME_SKILLS)),
        "project_slug": None,
    }


def _slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value.lower())
    return cleaned.strip("-") or "project"


def scope_id(scope: dict[str, Any]) -> str:
    """Stable identity for the scope, used as part of every job key."""
    kind = scope.get("kind")
    if kind not in KINDS:
        raise ScopeError(f"unknown scope kind: {kind!r}")
    if kind == KIND_NONE:
        return "none"
    return f"{kind}:{scope.get('root')}"


def trusted_dirs(home: Any = None) -> tuple[str, ...]:
    """Read ``skills.trusted_project_dirs``; project skills stay untrusted by default."""
    path = _home(home) / "config.yaml"
    if not path.is_file():
        return ()
    try:
        import yaml
    except Exception:  # pragma: no cover - the Hermes runtime ships PyYAML
        return ()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return ()
    skills = data.get("skills") or {}
    raw = skills.get("trusted_project_dirs") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    resolved = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            continue
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            candidate = _home(home) / candidate
        resolved.append(str(candidate.resolve()))
    return tuple(resolved)


def assert_write_scope(scope: dict[str, Any], target: Any, *, home: Any = None) -> Path:
    """A project write must stay inside the resolved project's skills root."""
    if scope.get("kind") != KIND_PROJECT:
        raise ScopeError("automatic project-skill writes require a resolved project scope")
    root = Path(scope["skills_root"]).resolve()
    path = Path(target).expanduser().resolve()
    if path != root and root not in path.parents:
        raise ScopeError(f"target escapes the project skills root: {path}")
    trusted = trusted_dirs(home)
    project_root = str(Path(scope["root"]).resolve())
    if project_root not in trusted:
        raise ScopeError(
            "project is not listed in skills.trusted_project_dirs; refusing an automatic write"
        )
    return path


def assert_target(scope: dict[str, Any], target: Any, *, home: Any = None) -> Path:
    """Containment for both scopes; a project scope additionally needs trust."""
    kind = scope.get("kind")
    if kind not in (KIND_PROJECT, KIND_HOME):
        raise ScopeError(f"scope {kind!r} cannot receive a skill write")
    root = Path(scope["skills_root"]).resolve()
    path = Path(target).expanduser().resolve()
    if path != root and root not in path.parents:
        raise ScopeError(f"target escapes the {kind} skills root: {path}")
    if kind == KIND_PROJECT:
        trusted = trusted_dirs(home)
        project_root = str(Path(scope["root"]).resolve())
        if project_root not in trusted:
            raise ScopeError(
                "project is not listed in skills.trusted_project_dirs; refusing an automatic write"
            )
    return path


def context_digest(context: dict[str, Any], *, keys: tuple[str, ...] | None = None) -> str:
    """Digest of the bounded decision context, ignoring free-form prose."""
    if not isinstance(context, dict):
        raise ScopeError("context must be an object")
    selected = keys or tuple(sorted(context))
    payload = {key: context.get(key) for key in selected}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def summarize(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": scope.get("kind"),
        "root": scope.get("root"),
        "skills_root": scope.get("skills_root"),
        "project_slug": scope.get("project_slug"),
        "scope_id": scope_id(scope),
    }
