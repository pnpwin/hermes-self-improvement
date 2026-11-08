from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hermes_self_improvement import skill_delta as sd


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "skill"
    (root / "references").mkdir(parents=True)
    (root / "SKILL.md").write_text("# Skill\n\nAlways verify.\n", encoding="utf-8")
    (root / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
    return root


def _patch(text: str = "Always verify.", new: str = "Always verify twice.") -> dict:
    return {"op": "patch_text", "path": "SKILL.md", "old_string": text, "new_string": new}


def test_valid_delta_applies_and_changes_digest(tmp_path):
    root = _package(tmp_path)
    before = sd.package_digest(root)
    delta = sd.validate_delta({"base_digest": before, "ops": [_patch()]}, base_digest=before, package_root=root)
    result = sd.apply_delta(delta, package_root=root)
    assert result["digest"] != before
    assert "verify twice" in (root / "SKILL.md").read_text(encoding="utf-8")
    assert sd.package_digest(root) == result["digest"]


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../escape.md", "references/../../x.md", "references//x.md", "references\\x.md", "a" * 2 + "/../b.md", ".hidden/x.md"],
)
def test_traversal_and_absolute_paths_are_rejected(path):
    with pytest.raises(sd.DeltaError):
        sd.validate_delta(
            {"base_digest": "d", "ops": [{"op": "write_support_file", "path": path, "content": "x"}]},
            base_digest="d",
        )


def test_digest_mismatch_is_rejected():
    with pytest.raises(sd.DeltaError):
        sd.validate_delta({"base_digest": "old", "ops": [_patch()]}, base_digest="new")


def test_patch_text_outside_skill_and_references_is_rejected():
    with pytest.raises(sd.DeltaError):
        sd.validate_delta(
            {"base_digest": "d", "ops": [{"op": "patch_text", "path": "src/plugin.py", "old_string": "a", "new_string": "b"}]},
            base_digest="d",
        )


def test_support_file_must_live_in_a_support_dir():
    with pytest.raises(sd.DeltaError):
        sd.validate_delta(
            {"base_digest": "d", "ops": [{"op": "write_support_file", "path": "bin/run.sh", "content": "x"}]},
            base_digest="d",
        )
    ok = sd.validate_delta(
        {"base_digest": "d", "ops": [{"op": "write_support_file", "path": "scripts/check.py", "content": "print(1)"}]},
        base_digest="d",
    )
    assert ok["ops"][0]["path"] == "scripts/check.py"


def test_op_count_and_file_count_and_size_are_capped():
    with pytest.raises(sd.DeltaError):
        sd.validate_delta({"base_digest": "d", "ops": [_patch()] * (sd.MAX_OPS + 1)}, base_digest="d")
    paths = ["references/a.md", "references/b.md", "references/c.md", "references/d.md"]
    ops = [
        {"op": "write_support_file", "path": path, "content": "x"} for path in paths
    ]
    with pytest.raises(sd.DeltaError):
        sd.validate_delta({"base_digest": "d", "ops": ops}, base_digest="d")
    with pytest.raises(sd.DeltaError):
        sd.validate_delta(
            {"base_digest": "d", "ops": [{"op": "write_support_file", "path": "assets/a.md", "content": "x" * (sd.MAX_CHANGED_CHARS + 1)}]},
            base_digest="d",
        )


def test_noop_and_bad_shapes_are_rejected():
    with pytest.raises(sd.DeltaError):
        sd.validate_delta({"base_digest": "d", "ops": [{"op": "patch_text", "path": "SKILL.md", "old_string": "a", "new_string": "a"}]}, base_digest="d")
    with pytest.raises(sd.DeltaError):
        sd.validate_delta({"base_digest": "d"}, base_digest="d")
    with pytest.raises(sd.DeltaError):
        sd.validate_delta([], base_digest="d")
    with pytest.raises(sd.DeltaError):
        sd.validate_delta({"base_digest": "d", "ops": []}, base_digest="d")
    with pytest.raises(sd.DeltaError):
        sd.validate_delta({"base_digest": "d", "ops": [{"op": "rewrite_all"}]}, base_digest="d")


def test_ambiguous_old_string_requires_replace_all(tmp_path):
    root = _package(tmp_path)
    (root / "SKILL.md").write_text("same\nsame\n", encoding="utf-8")
    digest = sd.package_digest(root)
    with pytest.raises(sd.DeltaError):
        sd.apply_delta(
            sd.validate_delta(
                {"base_digest": digest, "ops": [{"op": "patch_text", "path": "SKILL.md", "old_string": "same", "new_string": "other"}]},
                base_digest=digest,
                package_root=root,
            ),
            package_root=root,
        )
    all_delta = sd.validate_delta(
        {"base_digest": digest, "ops": [{"op": "patch_text", "path": "SKILL.md", "old_string": "same", "new_string": "other", "replace_all": True}]},
        base_digest=digest,
        package_root=root,
    )
    sd.apply_delta(all_delta, package_root=root)
    assert (root / "SKILL.md").read_text(encoding="utf-8") == "other\nother\n"


def test_missing_old_string_is_an_error(tmp_path):
    root = _package(tmp_path)
    digest = sd.package_digest(root)
    delta = sd.validate_delta(
        {"base_digest": digest, "ops": [{"op": "patch_text", "path": "SKILL.md", "old_string": "absent", "new_string": "x"}]},
        base_digest=digest,
        package_root=root,
    )
    with pytest.raises(sd.DeltaError):
        sd.apply_delta(delta, package_root=root)


def test_remove_requires_matching_digest(tmp_path):
    root = _package(tmp_path)
    digest = sd.package_digest(root)
    wrong = sd.validate_delta(
        {"base_digest": digest, "ops": [{"op": "remove_support_file", "path": "references/notes.md", "expected_sha256": "0" * 64}]},
        base_digest=digest,
    )
    with pytest.raises(sd.DeltaError):
        sd.apply_delta(wrong, package_root=root)
    real = hashlib.sha256((root / "references" / "notes.md").read_bytes()).hexdigest()
    good = sd.validate_delta(
        {"base_digest": sd.package_digest(root), "ops": [{"op": "remove_support_file", "path": "references/notes.md", "expected_sha256": real}]},
        base_digest=sd.package_digest(root),
    )
    sd.apply_delta(good, package_root=root)
    assert not (root / "references" / "notes.md").exists()


def test_symlink_escape_is_refused(tmp_path):
    root = _package(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "references" / "link").symlink_to(outside, target_is_directory=True)
    digest = sd.package_digest(root)
    with pytest.raises(sd.DeltaError):
        sd.validate_delta(
            {"base_digest": digest, "ops": [{"op": "write_support_file", "path": "references/link/x.md", "content": "x"}]},
            base_digest=digest,
            package_root=root,
        )


def test_secret_screen_detects_common_credentials():
    assert sd.secret_hits("no secrets here") == []
    for text in (
        "-----BEGIN RSA PRIVATE KEY-----",
        "ghp_" + "a" * 30,
        "github_pat_" + "b" * 30,
        'api_key = "abcdefghijklmnopqrst"',
        "Authorization: Bearer abcdefghijklmnop",
        "sk-" + "c" * 30,
    ):
        assert sd.secret_hits(text), text


def test_combine_requires_one_base_revision():
    first = {"base_digest": "d", "ops": [_patch()]}
    second = {"base_digest": "other", "ops": [_patch()]}
    with pytest.raises(sd.DeltaError):
        sd.combine_deltas([first, second], base_digest="d")
    merged = sd.combine_deltas([first], base_digest="d")
    assert len(merged["ops"]) == 1


def test_package_digest_is_order_independent_and_content_bound(tmp_path):
    root = _package(tmp_path)
    first = sd.package_digest(root)
    assert first == sd.package_digest(root)
    (root / "references" / "notes.md").write_text("changed\n", encoding="utf-8")
    assert sd.package_digest(root) != first
