"""The host-backed writer: the only path that may change a skill package.

A learning cycle never edits files itself. It hands a validated bounded delta to
this writer, which replays it through the host's own ``skill_manage`` tool so the
mutation keeps every native guard, lock, ledger and approval gate. Three rules
hold here:

* the base digest must still match, because a delta describes one revision;
* every patch must match before the first write happens, and the affected bytes
  are restored if the host refuses part-way, so a rejected op cannot leave a
  half-applied package;
* a staged or refused write is reported as not published, because an approval
  prompt is not a publication.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from . import skill_delta as sd

WRITER_NAME = "host_skill_tool"
_STAGE_KEYS = ("staged", "pending", "staged_for_approval", "awaiting_approval", "needs_approval")


class WriterError(RuntimeError):
    """Raised when the delta cannot be applied safely, before any write."""


def _load_manage() -> Callable[..., Any] | None:
    """Import the host skill tool; absence is a refusal, not an error to hide."""
    try:
        from tools.skill_manager_tool import skill_manage
    except Exception:
        return None
    return skill_manage if callable(skill_manage) else None


def diagnose() -> str | None:
    """Why the host writer is unavailable, or ``None`` when it is usable.

    ``_load_manage`` fails closed on any import error, which is right but silent:
    without this, a broken import looks exactly like a host that ships no skill
    tool, and every Standard/Heavy cycle reports the same bare "unavailable".
    """
    try:
        from tools.skill_manager_tool import skill_manage
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if not callable(skill_manage):
        return "host skill tool is not callable"
    return None


def _payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {"success": False, "error": "skill_manage returned non-json", "raw": raw[:400]}
        if isinstance(parsed, dict):
            return parsed
    return {"success": False, "error": "skill_manage returned an unexpected shape"}


def _preflight(root: Path, ops: list[dict[str, Any]]) -> None:
    """Check every match and digest before the first mutation."""
    for index, op in enumerate(ops):
        path = op["path"]
        if op["op"] == "patch_text":
            target = root / path
            if not target.is_file():
                raise WriterError(f"op {index}: {path} is missing from the package")
            text = target.read_text(encoding="utf-8")
            count = text.count(op["old_string"])
            if count == 0:
                raise WriterError(f"op {index}: old_string does not occur in {path}")
            if count > 1 and not op["replace_all"]:
                raise WriterError(f"op {index}: old_string occurs {count} times in {path}")
        elif op["op"] == "remove_support_file":
            target = root / path
            if not target.is_file():
                raise WriterError(f"op {index}: {path} is missing from the package")
            import hashlib

            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != op["expected_sha256"]:
                raise WriterError(f"op {index}: {path} changed since the delta was produced")


def _snapshot(root: Path, ops: list[dict[str, Any]]) -> dict[str, bytes | None]:
    """Capture the original bytes of every path the delta touches, once each."""
    saved: dict[str, bytes | None] = {}
    for op in ops:
        path = op["path"]
        if path in saved:
            continue
        target = root / path
        saved[path] = target.read_bytes() if target.is_file() else None
    return saved


def _restore(root: Path, saved: dict[str, bytes | None]) -> bool:
    """Put the touched paths back exactly; report whether the base was restored."""
    for path, original in saved.items():
        target = root / path
        if original is None:
            if target.exists():
                target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original)
    return True


def _call(manage: Callable[..., Any], name: str, op: dict[str, Any]) -> dict[str, Any]:
    kind = op["op"]
    if kind == "patch_text":
        raw = manage(
            action="patch",
            name=name,
            file_path=op["path"],
            old_string=op["old_string"],
            new_string=op["new_string"],
            replace_all=op["replace_all"],
        )
    elif kind == "write_support_file":
        raw = manage(action="write_file", name=name, file_path=op["path"], file_content=op["content"])
    else:
        raw = manage(action="remove_file", name=name, file_path=op["path"])
    return _payload(raw)


def build_writer(*, manage: Callable[..., Any] | None = None) -> Callable[..., dict[str, Any]] | None:
    """Return a writer bound to the host tool, or ``None`` when it is unavailable."""
    tool = manage if callable(manage) else _load_manage()
    if not callable(tool):
        return None

    def writer(*, delta: dict[str, Any], target_root: Any, base_digest: str) -> dict[str, Any]:
        root = Path(target_root).resolve()
        if not root.is_dir():
            raise WriterError(f"target package is not a directory: {root}")
        if sd.package_digest(root) != base_digest:
            return {
                "published": False,
                "writer": WRITER_NAME,
                "reason": "base_digest_drift",
                "applied_ops": 0,
            }
        # The lane hands over its own validated form, which also carries
        # bookkeeping such as ``changed_characters``. Re-validate the structural
        # pair so the caps and containment are re-checked against this package,
        # without rejecting the lane's own normalized shape.
        structural = {"base_digest": delta.get("base_digest"), "ops": delta.get("ops")}
        clean = sd.validate_delta(structural, base_digest=base_digest, package_root=root)
        _preflight(root, clean["ops"])
        saved = _snapshot(root, clean["ops"])
        name = root.name
        applied = 0

        def _stop(reason: str, index: int, error: Any = None) -> dict[str, Any]:
            # The host can still refuse or stage a write after earlier ops landed.
            # Preflight cannot predict that, so undo what did land and say so.
            rolled = applied > 0
            if rolled:
                _restore(root, saved)
            outcome: dict[str, Any] = {
                "published": False,
                "writer": WRITER_NAME,
                "reason": reason,
                "applied_ops": applied,
                "failed_op": index,
                "rolled_back": rolled,
                "digest_restored": sd.package_digest(root) == base_digest,
            }
            if error is not None:
                outcome["error"] = error
            return outcome

        for index, op in enumerate(clean["ops"]):
            result = _call(tool, name, op)
            if any(result.get(key) for key in _STAGE_KEYS):
                return _stop("staged_for_approval", index)
            if result.get("success") is not True:
                return _stop(
                    "write_refused", index, result.get("error") or result.get("message")
                )
            applied += 1
        return {
            "published": True,
            "writer": WRITER_NAME,
            "applied_ops": applied,
            "digest": sd.package_digest(root),
        }

    return writer
