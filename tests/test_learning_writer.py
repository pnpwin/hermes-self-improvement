"""The host-backed writer: the only path that may change a skill package."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_self_improvement import learning_writer as lw
from hermes_self_improvement import skill_delta as sd


def _package(root: Path, body: str = "line one\nline two\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        "---\nname: demo\ndescription: \"Use when demoing.\"\nversion: 1.0.0\n---\n\n" + body,
        encoding="utf-8",
    )
    (root / "references").mkdir(exist_ok=True)
    (root / "references" / "eval.md").write_text("old detail\n", encoding="utf-8")
    return root


def _delta(root: Path, ops: list[dict]) -> dict:
    return {"base_digest": sd.package_digest(root), "ops": ops}


class FakeTool:
    """Records the calls, optionally refuses or stages one of them."""

    def __init__(
        self,
        root: Path,
        *,
        fail_at: int | None = None,
        staged: bool = False,
        stage_at: int | None = None,
    ):
        self.root = root
        self.fail_at = fail_at
        self.staged = staged
        self.stage_at = stage_at
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if self.staged or (self.stage_at is not None and index == self.stage_at):
            return json.dumps({"success": True, "staged": True, "message": "awaiting approval"})
        if self.fail_at is not None and index == self.fail_at:
            return json.dumps({"success": False, "error": "patch did not match"})
        target = self.root / kwargs["file_path"] if kwargs.get("file_path") else self.root / "SKILL.md"
        if kwargs["action"] == "patch":
            text = target.read_text(encoding="utf-8")
            assert kwargs["old_string"] in text
            target.write_text(text.replace(kwargs["old_string"], kwargs["new_string"], 1), encoding="utf-8")
        elif kwargs["action"] == "write_file":
            (self.root / kwargs["file_path"]).write_text(kwargs["file_content"], encoding="utf-8")
        elif kwargs["action"] == "remove_file":
            (self.root / kwargs["file_path"]).unlink()
        return json.dumps({"success": True})


def test_writer_applies_every_op_and_reaches_the_candidate_digest(tmp_path):
    root = _package(tmp_path / "demo")
    delta = _delta(
        root,
        [
            {"op": "patch_text", "path": "SKILL.md", "old_string": "line two", "new_string": "line 2"},
            {"op": "write_support_file", "path": "references/eval.md", "content": "new detail\n"},
        ],
    )
    expected = tmp_path / "expected"
    import shutil

    shutil.copytree(root, expected)
    sd.apply_delta(sd.validate_delta(delta, base_digest=delta["base_digest"], package_root=expected), package_root=expected)

    tool = FakeTool(root)
    writer = lw.build_writer(manage=tool)
    assert writer is not None
    result = writer(delta=delta, target_root=str(root), base_digest=delta["base_digest"])

    assert result["published"] is True
    assert result["applied_ops"] == 2
    assert result["digest"] == sd.package_digest(expected)
    assert [call["action"] for call in tool.calls] == ["patch", "write_file"]
    assert tool.calls[0]["file_path"] == "SKILL.md"
    assert tool.calls[0]["name"] == "demo"


def test_writer_refuses_on_digest_drift_without_touching_the_package(tmp_path):
    root = _package(tmp_path / "demo")
    delta = _delta(root, [{"op": "patch_text", "path": "SKILL.md", "old_string": "line two", "new_string": "x"}])
    (root / "SKILL.md").write_text("drifted\n", encoding="utf-8")

    tool = FakeTool(root)
    result = lw.build_writer(manage=tool)(
        delta=delta, target_root=str(root), base_digest=delta["base_digest"]
    )

    assert result["published"] is False
    assert result["reason"] == "base_digest_drift"
    assert tool.calls == []


def test_writer_refuses_a_staged_write_instead_of_claiming_success(tmp_path):
    root = _package(tmp_path / "demo")
    delta = _delta(root, [{"op": "patch_text", "path": "SKILL.md", "old_string": "line two", "new_string": "x"}])

    result = lw.build_writer(manage=FakeTool(root, staged=True))(
        delta=delta, target_root=str(root), base_digest=delta["base_digest"]
    )

    assert result["published"] is False
    assert result["reason"] == "staged_for_approval"
    assert result["rolled_back"] is False  # nothing landed, so nothing to undo
    assert result["digest_restored"] is True
    assert sd.package_digest(root) == delta["base_digest"]


def test_a_missing_old_string_is_caught_before_any_write(tmp_path):
    root = _package(tmp_path / "demo")
    delta = _delta(
        root,
        [
            {"op": "patch_text", "path": "SKILL.md", "old_string": "line 2", "new_string": "kept"},
            {"op": "patch_text", "path": "SKILL.md", "old_string": "absent text", "new_string": "x"},
        ],
    )

    tool = FakeTool(root)
    with pytest.raises(lw.WriterError, match="does not occur"):
        lw.build_writer(manage=tool)(
            delta=delta, target_root=str(root), base_digest=delta["base_digest"]
        )
    assert tool.calls == []


def test_a_refused_op_restores_the_package_it_already_touched(tmp_path):
    root = _package(tmp_path / "demo")
    delta = _delta(
        root,
        [
            {"op": "patch_text", "path": "SKILL.md", "old_string": "line one", "new_string": "line 1"},
            {"op": "patch_text", "path": "SKILL.md", "old_string": "line two", "new_string": "line 2"},
        ],
    )

    result = lw.build_writer(manage=FakeTool(root, fail_at=1))(
        delta=delta, target_root=str(root), base_digest=delta["base_digest"]
    )

    assert result["published"] is False
    assert result["reason"] == "write_refused"
    assert result["applied_ops"] == 1
    assert result["failed_op"] == 1
    assert result["rolled_back"] is True
    assert result["digest_restored"] is True
    # The op that landed is undone: the package is the revision the gate assessed.
    assert sd.package_digest(root) == delta["base_digest"]
    assert "line one" in (root / "SKILL.md").read_text(encoding="utf-8")


def test_removal_requires_the_recorded_digest(tmp_path):
    root = _package(tmp_path / "demo")
    digest = __import__("hashlib").sha256((root / "references/eval.md").read_bytes()).hexdigest()
    wrong = _delta(
        root,
        [
            {
                "op": "remove_support_file",
                "path": "references/eval.md",
                "expected_sha256": "0" * 64,
            }
        ],
    )
    tool = FakeTool(root)
    with pytest.raises(lw.WriterError, match="changed since the delta was produced"):
        lw.build_writer(manage=tool)(
            delta=wrong, target_root=str(root), base_digest=wrong["base_digest"]
        )
    assert tool.calls == []

    right = _delta(
        root,
        [{"op": "remove_support_file", "path": "references/eval.md", "expected_sha256": digest}],
    )
    result = lw.build_writer(manage=tool)(
        delta=right, target_root=str(root), base_digest=right["base_digest"]
    )
    assert result["published"] is True
    assert not (root / "references/eval.md").exists()


def test_no_host_tool_means_no_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(lw, "_load_manage", lambda: None)
    assert lw.build_writer() is None


def test_a_stage_after_one_op_also_restores_the_package(tmp_path):
    """A deferred write must not leave the first op behind either."""
    root = _package(tmp_path / "demo")
    before = sd.package_digest(root)
    delta = _delta(
        root,
        [
            {"op": "patch_text", "path": "SKILL.md", "old_string": "line one", "new_string": "line 1"},
            {"op": "patch_text", "path": "SKILL.md", "old_string": "line two", "new_string": "line 2"},
        ],
    )

    result = lw.build_writer(manage=FakeTool(root, stage_at=1))(
        delta=delta, target_root=str(root), base_digest=delta["base_digest"]
    )

    assert result["reason"] == "staged_for_approval"
    assert result["applied_ops"] == 1
    assert result["rolled_back"] is True
    assert result["digest_restored"] is True
    assert sd.package_digest(root) == before
    assert "line one" in (root / "SKILL.md").read_text(encoding="utf-8")


def test_a_created_support_file_is_removed_again_on_rollback(tmp_path):
    """Rollback must not leave a file the delta created."""
    root = _package(tmp_path / "demo")
    before = sd.package_digest(root)
    delta = _delta(
        root,
        [
            {"op": "write_support_file", "path": "references/new.md", "content": "fresh"},
            {"op": "patch_text", "path": "SKILL.md", "old_string": "line two", "new_string": "line 2"},
        ],
    )

    result = lw.build_writer(manage=FakeTool(root, fail_at=1))(
        delta=delta, target_root=str(root), base_digest=delta["base_digest"]
    )

    assert result["rolled_back"] is True
    assert not (root / "references/new.md").exists()
    assert sd.package_digest(root) == before
