"""Bounded, reviewable skill deltas.

The Curator never rewrites a package. It returns a small list of typed
operations, and every operation is validated here before anything is written:
no traversal, no symlink escape, no absolute paths, no unbounded rewrite, and a
hard cap on how much text one job may change.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

SKILL_FILE = "SKILL.md"
SUPPORT_DIRS = ("references", "templates", "scripts", "assets")
MAX_OPS = 8
MAX_TEXT_FILES = 3
MAX_CHANGED_CHARS = 8000
TEXT_SUFFIXES = (".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".py", ".sh")

_OP_KINDS = ("patch_text", "write_support_file", "remove_support_file")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)\bauthorization:\s*bearer\s+\S{10,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
)


class DeltaError(ValueError):
    """Raised when a delta is malformed or exceeds the bounded-change contract."""


def _relative_path(value: Any, *, field: str = "path") -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DeltaError(f"{field} must be a non-empty relative path")
    if _CONTROL.search(value):
        raise DeltaError(f"{field} contains a control character")
    if value.startswith("/") or value.startswith("\\"):
        raise DeltaError(f"{field} must be relative: {value}")
    if "\\" in value:
        raise DeltaError(f"{field} must use forward slashes: {value}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise DeltaError(f"{field} contains an empty, dot or parent segment: {value}")
    if len(parts) > 4:
        raise DeltaError(f"{field} is nested too deeply: {value}")
    return value


def _is_skill_text(path: str) -> bool:
    return path == SKILL_FILE or (path.startswith("references/") and path.endswith(".md"))


def _is_support_file(path: str) -> bool:
    head = path.split("/", 1)[0]
    return "/" in path and head in SUPPORT_DIRS and path.endswith(TEXT_SUFFIXES)


def _text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise DeltaError(f"{field} must be a string")
    if not allow_empty and not value:
        raise DeltaError(f"{field} must not be empty")
    if _CONTROL.search(value):
        raise DeltaError(f"{field} contains a control character")
    return value


def secret_hits(text: str) -> list[str]:
    """Cheap, deterministic secret screen; it never claims to be a full scanner."""
    if not isinstance(text, str):
        return []
    return [pattern.pattern for pattern in _SECRET_PATTERNS if pattern.search(text)]


def validate_delta(delta: Any, *, base_digest: str, package_root: Any = None) -> dict[str, Any]:
    if not isinstance(delta, dict):
        raise DeltaError("delta must be an object")
    if set(delta) != {"base_digest", "ops"}:
        raise DeltaError("delta requires exactly base_digest and ops")
    if delta["base_digest"] != base_digest:
        raise DeltaError("delta was produced for another package revision")
    ops = delta["ops"]
    if not isinstance(ops, list):
        raise DeltaError("ops must be a list")
    if not 1 <= len(ops) <= MAX_OPS:
        raise DeltaError(f"ops must contain 1..{MAX_OPS} operations")
    normalized: list[dict[str, Any]] = []
    text_files: set[str] = set()
    changed = 0
    root = Path(package_root).resolve() if package_root is not None else None
    for index, raw in enumerate(ops):
        if not isinstance(raw, dict):
            raise DeltaError(f"op {index} must be an object")
        kind = raw.get("op")
        if kind not in _OP_KINDS:
            raise DeltaError(f"op {index} has an unsupported kind")
        if kind == "patch_text":
            path = _relative_path(raw.get("path"), field=f"op {index} path")
            if not _is_skill_text(path):
                raise DeltaError(f"op {index}: patch_text targets only {SKILL_FILE} or references/*.md")
            old = _text(raw.get("old_string"), f"op {index} old_string")
            new = _text(raw.get("new_string"), f"op {index} new_string", allow_empty=True)
            if old == new:
                raise DeltaError(f"op {index} is a no-op")
            replace_all = raw.get("replace_all", False)
            if type(replace_all) is not bool:
                raise DeltaError(f"op {index} replace_all must be a boolean")
            text_files.add(path)
            changed += max(len(old), len(new))
            normalized.append(
                {
                    "op": "patch_text",
                    "path": path,
                    "old_string": old,
                    "new_string": new,
                    "replace_all": replace_all,
                }
            )
        elif kind == "write_support_file":
            path = _relative_path(raw.get("path"), field=f"op {index} path")
            if not _is_support_file(path):
                raise DeltaError(f"op {index}: support files live under {', '.join(SUPPORT_DIRS)}")
            content = _text(raw.get("content"), f"op {index} content", allow_empty=True)
            changed += len(content)
            text_files.add(path)
            normalized.append({"op": "write_support_file", "path": path, "content": content})
        else:
            path = _relative_path(raw.get("path"), field=f"op {index} path")
            if not _is_support_file(path):
                raise DeltaError(f"op {index}: support files live under {', '.join(SUPPORT_DIRS)}")
            expected = raw.get("expected_sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise DeltaError(f"op {index} expected_sha256 must be a lowercase sha256")
            text_files.add(path)
            normalized.append(
                {"op": "remove_support_file", "path": path, "expected_sha256": expected}
            )
    if len(text_files) > MAX_TEXT_FILES:
        raise DeltaError(f"a delta may change at most {MAX_TEXT_FILES} files")
    if changed > MAX_CHANGED_CHARS:
        raise DeltaError(f"a delta may change at most {MAX_CHANGED_CHARS} characters")
    if root is not None:
        _assert_inside(root, normalized)
    return {"base_digest": base_digest, "ops": normalized, "changed_characters": changed}


def _assert_inside(root: Path, ops: list[dict[str, Any]]) -> None:
    for item in ops:
        target = (root / item["path"]).resolve()
        if root not in target.parents and target != root:
            raise DeltaError(f"op escapes the package root: {item['path']}")
        if any(parent.is_symlink() for parent in target.parents if root in parent.parents):
            raise DeltaError(f"op crosses a symlink: {item['path']}")


def package_digest(root: Any) -> str:
    base = Path(root).resolve()
    if not base.is_dir():
        raise DeltaError(f"package root is not a directory: {base}")
    digest = hashlib.sha256()
    for path in sorted(base.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(base).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def apply_delta(delta: dict[str, Any], *, package_root: Any) -> dict[str, Any]:
    """Apply a validated delta to a staging copy and return its new digest."""
    root = Path(package_root).resolve()
    for item in delta["ops"]:
        target = (root / item["path"]).resolve()
        if root not in target.parents and target != root:
            raise DeltaError(f"op escapes the package root: {item['path']}")
        if item["op"] == "patch_text":
            if not target.is_file():
                raise DeltaError(f"patch target is missing: {item['path']}")
            text = target.read_text(encoding="utf-8")
            occurrences = text.count(item["old_string"])
            if occurrences == 0:
                raise DeltaError(f"old_string not found in {item['path']}")
            if occurrences > 1 and not item["replace_all"]:
                raise DeltaError(f"old_string is ambiguous in {item['path']}")
            updated = text.replace(item["old_string"], item["new_string"]) if item["replace_all"] else text.replace(
                item["old_string"], item["new_string"], 1
            )
            target.write_text(updated, encoding="utf-8")
        elif item["op"] == "write_support_file":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item["content"], encoding="utf-8")
        else:
            if not target.is_file():
                raise DeltaError(f"remove target is missing: {item['path']}")
            current = hashlib.sha256(target.read_bytes()).hexdigest()
            if current != item["expected_sha256"]:
                raise DeltaError(f"remove target changed since the delta was produced: {item['path']}")
            target.unlink()
    return {"digest": package_digest(root), "changed_paths": [item["path"] for item in delta["ops"]]}


def combine_deltas(deltas: list[dict[str, Any]], *, base_digest: str) -> dict[str, Any]:
    """Concatenate validated deltas, re-validating the merged result once."""
    merged = {"base_digest": base_digest, "ops": []}
    for delta in deltas:
        if delta.get("base_digest") != base_digest:
            raise DeltaError("cannot combine deltas from different revisions")
        merged["ops"].extend(delta["ops"])
    return validate_delta(merged, base_digest=base_digest)


def summarize(delta: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_digest": delta.get("base_digest"),
        "operations": len(delta.get("ops") or []),
        "changed_characters": delta.get("changed_characters"),
        "paths": sorted({item.get("path") for item in delta.get("ops") or []}),
    }
