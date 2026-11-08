from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_self_improvement import learning_runtime as rt
from hermes_self_improvement import skill_delta as sd

ROUTE = {"provider": "p", "model": "m", "reasoning": "low"}
LESSON = {
    "observation": "trailing empty fields were dropped",
    "hypothesis": "split on every tab",
    "evidence_ids": ["t1"],
    "confidence": "medium",
    "scope": "home",
}


def _home(tmp_path: Path, *, modes: dict | None = None, routes: bool = True) -> Path:
    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    settings = {
        "lite_enabled": True,
        "standard_enabled": False,
        "heavy_enabled": False,
    }
    settings.update(modes or {})
    lines = ["plugins:", "  entries:", "    hermes-skillcraft:", "      settings:"]
    for key, value in settings.items():
        lines.append(f"        {key}: {'true' if value else 'false'}")
    if routes:
        lines.append("        learning_routes:")
        for role in ("reflector", "curator", "task", "reviewer"):
            lines.append(f"          {role}:")
            for field, item in ROUTE.items():
                lines.append(f"            {field}: {item}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return home


def _package(home: Path) -> Path:
    root = home / ".agents" / "skills" / "demo"
    (root / "references").mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        "---\nname: demo\ndescription: \"Use when splitting tabs.\"\nversion: 1.0.0\n---\n"
        "# Demo\n\nAlways verify. See [notes](references/notes.md).\n",
        encoding="utf-8",
    )
    (root / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
    return root


def test_status_is_read_only(tmp_path):
    home = _home(tmp_path)
    result = rt.run_action({"action": "status"}, home=home)
    assert result["ok"] is True
    assert result["model_calls"] == 0
    assert result["policy"]["pipeline"] == "lite"
    assert result["budget"]["used_model_calls"] == 0


def test_check_reports_structure_without_a_model_call(tmp_path):
    home = _home(tmp_path)
    root = _package(home)
    result = rt.run_action({"action": "check", "package": str(root)}, home=home)
    assert result["ok"] is True
    assert result["skill"]["name"] == "demo"
    assert result["skill"]["missing_links"] == []
    assert result["skill"]["digest"] == sd.package_digest(root)
    assert result["model_calls"] == 0


def test_check_flags_a_missing_support_link(tmp_path):
    home = _home(tmp_path)
    root = _package(home)
    (root / "references" / "notes.md").unlink()
    result = rt.run_action({"action": "check", "package": str(root)}, home=home)
    assert result["ok"] is False
    assert result["skill"]["missing_links"] == ["references/notes.md"]


def test_lesson_without_authorization_makes_no_call(tmp_path):
    home = _home(tmp_path)
    root = _package(home)
    result = rt.run_action({"action": "lesson", "package": str(root)}, home=home)
    assert result["ok"] is False
    assert "allow_model_calls" in result["error"]


def test_lesson_without_routes_fails_closed(tmp_path):
    home = _home(tmp_path, routes=False)
    root = _package(home)
    result = rt.run_action(
        {"action": "lesson", "package": str(root), "allow_model_calls": True}, home=home
    )
    assert result["ok"] is False
    assert result["status"] == "route_unconfigured"


def test_lesson_with_injected_dependencies_saves_one_lesson(tmp_path, monkeypatch):
    home = _home(tmp_path)
    root = _package(home)
    before = sd.package_digest(root)

    def fake_deps(_home, *, package_root, cases_path):
        return {
            "reflector": lambda payload, route: {"text": json.dumps(LESSON), "usage": {"input_tokens": 1}},
            "package_root": package_root,
        }

    monkeypatch.setattr(rt, "_build_deps", fake_deps)
    result = rt.run_action(
        {"action": "lesson", "package": str(root), "allow_model_calls": True, "cycle_id": "c-1"},
        home=home,
    )
    assert result["ok"] is True
    assert result["status"] == "lesson_saved"
    assert result["model_calls"] == 1
    assert sd.package_digest(root) == before
    report = rt.run_action({"action": "report"}, home=home)
    assert report["lessons"][0]["job_id"] == "c-1"


def test_missing_package_is_refused_before_a_model_call(tmp_path):
    home = _home(tmp_path)
    result = rt.run_action(
        {"action": "lesson", "package": str(tmp_path / "absent"), "allow_model_calls": True}, home=home
    )
    assert result["ok"] is False
    assert result["error"] == "skill package is not loadable"


def test_exposure_round_trip(tmp_path):
    home = _home(tmp_path)
    digest = sd.package_digest(_package(home))
    record = rt.run_action(
        {
            "action": "exposure",
            "record": {
                "event_id": "e1",
                "skill_name": "demo",
                "skill_digest": digest,
                "scope_id": "home:/h",
                "session_id": "s1",
            },
        },
        home=home,
    )
    assert record["duplicate"] is False
    bound = rt.run_action(
        {
            "action": "exposure",
            "bind": {
                "exposure_id": record["exposure_id"],
                "outcome_id": "o1",
                "kind": "task_passed",
                "passed": True,
                "evidence_ids": ["t1"],
                "expected_digest": digest,
            },
        },
        home=home,
    )
    assert bound["bound"] is True
    summary = rt.run_action({"action": "exposure", "skill_name": "demo"}, home=home)
    assert summary["versions"][0]["passed"] == 1


def test_unknown_action_is_rejected(tmp_path):
    home = _home(tmp_path)
    assert rt.run_action({"action": "train"}, home=home)["ok"] is False
    assert rt.run_action({}, home=home)["ok"] is False
