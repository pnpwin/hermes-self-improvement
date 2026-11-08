from __future__ import annotations

from pathlib import Path

import pytest

from hermes_self_improvement import skill_scope as ss


def _git(path: Path) -> Path:
    (path / ".git").mkdir(parents=True)
    return path


def test_home_cwd_resolves_to_home_scope(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    scope = ss.resolve_scope(home, home)
    assert scope["kind"] == ss.KIND_HOME
    assert scope["skills_root"] == str(home / ".agents" / "skills")
    assert scope["project_slug"] is None


def test_nearest_git_project_wins_over_home(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    project = _git(tmp_path / "work" / "alpha")
    nested = project / "src" / "pkg"
    nested.mkdir(parents=True)
    scope = ss.resolve_scope(nested, home)
    assert scope["kind"] == ss.KIND_PROJECT
    assert scope["root"] == str(project.resolve())
    assert scope["project_slug"] == "alpha"
    assert scope["skills_root"] == str(project.resolve() / ".agents" / "skills")


def test_cwd_outside_any_project_falls_back_to_home_scope(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    loose = tmp_path / "loose"
    loose.mkdir()
    scope = ss.resolve_scope(loose, home)
    assert scope["kind"] == ss.KIND_HOME


def test_missing_cwd_is_an_error(tmp_path):
    with pytest.raises(ss.ScopeError):
        ss.resolve_scope(None, tmp_path)
    with pytest.raises(ss.ScopeError):
        ss.resolve_scope(tmp_path / "absent", tmp_path)


def test_scope_id_is_stable_and_kind_prefixed(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    project = _git(tmp_path / "p")
    a = ss.resolve_scope(project, home)
    b = ss.resolve_scope(project, home)
    assert ss.scope_id(a) == ss.scope_id(b)
    assert ss.scope_id(a).startswith("project:")
    assert ss.scope_id({"kind": "home", "root": str(home)}) == f"home:{home.resolve()}"
    assert ss.scope_id({"kind": "none"}) == "none"
    with pytest.raises(ss.ScopeError):
        ss.scope_id({"kind": "space"})


def test_project_write_requires_trusted_dir(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    project = _git(tmp_path / "proj")
    scope = ss.resolve_scope(project, home)
    target = Path(scope["skills_root"]) / "new-skill" / "SKILL.md"
    with pytest.raises(ss.ScopeError):
        ss.assert_write_scope(scope, target, home=home)
    (home / "config.yaml").write_text(
        "skills:\n  trusted_project_dirs:\n"
        f"    - {project.resolve()}\n",
        encoding="utf-8",
    )
    assert ss.assert_write_scope(scope, target, home=home) == target.resolve()


def test_project_write_cannot_escape_the_project_skills_root(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    project = _git(tmp_path / "proj")
    (home / "config.yaml").write_text(
        f"skills:\n  trusted_project_dirs:\n    - {project.resolve()}\n", encoding="utf-8"
    )
    scope = ss.resolve_scope(project, home)
    with pytest.raises(ss.ScopeError):
        ss.assert_write_scope(scope, project / "src" / "SKILL.md", home=home)


def test_home_scope_refuses_automatic_project_write(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    scope = ss.resolve_scope(home, home)
    with pytest.raises(ss.ScopeError):
        ss.assert_write_scope(scope, home / ".agents" / "skills" / "x" / "SKILL.md", home=home)


def test_context_digest_is_order_insensitive_and_prose_free():
    first = {"a": 1, "b": ["x"], "prose": "long text one"}
    second = {"b": ["x"], "a": 1, "prose": "long text two"}
    assert ss.context_digest(first, keys=("a", "b")) == ss.context_digest(second, keys=("a", "b"))
    assert ss.context_digest(first) != ss.context_digest(second)
