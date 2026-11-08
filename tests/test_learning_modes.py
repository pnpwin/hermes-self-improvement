from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_self_improvement import learning_budget as lb
from hermes_self_improvement import learning_modes as lm
from hermes_self_improvement import learning_policy as lp
from hermes_self_improvement import learning_store as ls
from hermes_self_improvement import skill_delta as sd
from hermes_self_improvement import skill_scope as ss

ROUTE = {"provider": "p", "model": "m", "reasoning": "low"}
WHEN = datetime(2026, 11, 9, 10, 0, tzinfo=timezone.utc)
LESSON = {
    "observation": "the parser dropped trailing empty fields",
    "hypothesis": "splitting on every tab and keeping empties fixes it",
    "evidence_ids": ["t1"],
    "confidence": "medium",
    "scope": "project",
}
GOOD_COMPARISON = {
    "installed_passes": 3,
    "candidate_passes": 4,
    "critical_regressions": 0,
    "pairs": 4,
    "eligible": True,
}
GOOD_HOLDOUT = {"untouched": True, "pairs": 3, "installed_passes": 2, "candidate_passes": 3}
GOOD_CASES = {"trusted": 4, "total": 5}
CLEAN_SCAN = {"status": "pass", "findings": []}


def _policy(**flags) -> dict:
    routes = {role: dict(ROUTE) for role in lp.ROUTE_ROLES}
    return lp.build_policy({**flags, "learning_routes": routes})


def _package(root: Path) -> Path:
    (root / "references").mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text("# Skill\n\nAlways verify.\n", encoding="utf-8")
    (root / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
    return root


def _deps(
    tmp_path: Path,
    root: Path,
    *,
    lesson: dict | None = None,
    comparison: dict | None = None,
    holdout: dict | None = None,
    cases: dict | None = None,
    scan: dict | None = None,
    writer=None,
    delta_factory=None,
    review: dict | None = None,
):
    calls: list[dict] = []
    state = {"staging_index": 0}

    def reflector(payload, route):
        calls.append({"role": "reflector", "route": route, "prompt": payload["prompt"]})
        return {"text": json.dumps(lesson or LESSON), "usage": {"input_tokens": 10}}

    def curator(payload, route):
        calls.append({"role": "curator", "base_digest": payload["base_digest"]})
        base = payload["base_digest"]
        if delta_factory is not None:
            body = delta_factory(base)
        else:
            body = {
                "base_digest": base,
                "ops": [
                    {
                        "op": "patch_text",
                        "path": "SKILL.md",
                        "old_string": "Always verify.",
                        "new_string": "Always verify, twice.",
                    }
                ],
            }
        return {"text": json.dumps(body), "usage": {"output_tokens": 20}}

    def evaluator(payload, route):
        calls.append({"role": "task", "candidate_root": payload["candidate_root"]})
        return {
            "comparison": dict(comparison or GOOD_COMPARISON),
            "holdout": dict(holdout or GOOD_HOLDOUT),
            "cases": dict(cases or GOOD_CASES),
            "scan": dict(scan or CLEAN_SCAN),
        }

    def reviewer(payload, route):
        calls.append({"role": "reviewer", "route": route, "prompt": payload["prompt"]})
        body = review or {"verdict": "approve", "findings": []}
        return {"text": json.dumps(body), "usage": {"output_tokens": 5}}

    def staging(base_root):
        state["staging_index"] += 1
        target = tmp_path / f"staging-{state['staging_index']}"
        shutil.copytree(base_root, target)
        return target

    deps = {
        "reflector": reflector,
        "curator": curator,
        "evaluator": evaluator,
        "reviewer": reviewer,
        "staging": staging,
        "package_root": root,
        "calls": calls,
    }
    if writer is not None:
        deps["native_writer"] = writer
    return deps, calls, state


def _writer(root: Path):
    def write(*, delta, target_root, base_digest):
        sd.apply_delta(delta, package_root=Path(target_root))
        return {"published": True, "writer": "test-native"}

    return write


def _scope(tmp_path: Path) -> dict:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return ss.resolve_scope(home, home)


def _run(tmp_path, *, policy=None, deps, scope=None, job_id="job-1", evidence=None, dry_run=False, manifest=None, deadline=None, dedupe=False):
    scope = scope or _scope(tmp_path)
    ledger = lb.Ledger(tmp_path / "budget.sqlite3")
    store = ls.LessonStore(tmp_path / "lessons.sqlite3")
    result = lm.run_learning_cycle(
        policy=policy or _policy(),
        ledger=ledger,
        store=store,
        scope=scope,
        evidence=evidence or {"task": "split TSV", "logs": "L1: field count 2"},
        job_id=job_id,
        when=WHEN,
        deps=deps,
        dry_run=dry_run,
        search_manifest=manifest,
        deadline=deadline,
        dedupe=dedupe,
    )
    return result, ledger, store


# -- Lite ---------------------------------------------------------------


def test_lite_saves_one_lesson_and_changes_nothing(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)
    deps, calls, _ = _deps(tmp_path, root)
    result, ledger, store = _run(tmp_path, policy=_policy(lite_enabled=True, standard_enabled=False), deps=deps)
    assert result["status"] == "lesson_saved"
    assert result["mode"] == "lite"
    assert result["instruction_changed"] is False
    assert len(calls) == 1 and calls[0]["role"] == "reflector"
    assert sd.package_digest(root) == before
    assert store.lesson("job-1")["lesson"]["observation"].startswith("the parser")
    assert ledger.job("job-1")["outcome"] == "lesson_saved"
    assert ledger.status(lp.day_key(_policy(), WHEN), daily_cap=24, lite_daily_cap=8)["used_lite_calls"] == 1


def test_lite_keeps_an_imperative_observation_out_of_the_skill(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)
    deps, _, _ = _deps(tmp_path, root, lesson={**LESSON, "observation": "Always split on every tab."})
    result, _, _ = _run(tmp_path, policy=_policy(lite_enabled=True, standard_enabled=False), deps=deps)
    assert result["lesson"]["operational_instruction"] is True
    assert result["lesson"]["instruction_changed"] is False
    assert "Observation only" in result["lesson"]["note"]
    assert sd.package_digest(root) == before


def test_disabled_pipeline_makes_no_call(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root)
    policy = _policy(lite_enabled=False, standard_enabled=False, heavy_enabled=False)
    result, ledger, _ = _run(tmp_path, policy=policy, deps=deps)
    assert result["status"] == "learning_disabled"
    assert calls == []
    assert ledger.job("job-1") is None


def test_missing_route_fails_closed_before_any_job(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root)
    policy = lp.build_policy({"lite_enabled": True, "standard_enabled": False})
    result, ledger, _ = _run(tmp_path, policy=policy, deps=deps)
    assert result["status"] == "route_unconfigured"
    assert result["missing"] == ["reflector"]
    assert calls == []
    assert ledger.job("job-1") is None


def test_backend_unavailable_is_reported_without_calls(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root)
    deps.pop("reflector")
    result, _, _ = _run(tmp_path, policy=_policy(lite_enabled=True, standard_enabled=False), deps=deps)
    assert result["status"] == "backend_unavailable"
    assert calls == []


def test_dry_run_reserves_nothing(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root)
    result, ledger, _ = _run(tmp_path, policy=_policy(lite_enabled=True, standard_enabled=False), deps=deps, dry_run=True)
    assert result["status"] == "dry_run"
    assert result["ceiling"] == 1
    assert result["model_calls"] == 0
    assert calls == []
    assert ledger.job("job-1") is None


def test_lite_daily_cap_refuses_the_call_before_dispatch(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root)
    policy = _policy(lite_enabled=True, standard_enabled=False)
    policy["budget"]["lite_daily_model_calls"] = 1
    _run(tmp_path, policy=policy, deps=deps, job_id="job-a")
    result, _, _ = _run(tmp_path, policy=policy, deps=deps, job_id="job-b")
    assert result["status"] == "budget_denied"
    assert result["reason"] == "lite_daily_cap"
    assert len(calls) == 1


def test_invalid_lesson_json_stops_without_writing(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)
    deps, _, _ = _deps(tmp_path, root)

    def broken(payload, route):
        return {"text": "not json"}

    deps["reflector"] = broken
    result, ledger, store = _run(tmp_path, policy=_policy(lite_enabled=True, standard_enabled=False), deps=deps)
    assert result["status"] == "stopped_invalid_stage"
    assert store.lesson("job-1") is None
    assert sd.package_digest(root) == before
    assert ledger.job("job-1")["outcome"] == "stopped_invalid_stage"


def test_the_same_job_id_never_spends_a_second_budget(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root)
    policy = _policy(lite_enabled=True, standard_enabled=False)
    _run(tmp_path, policy=policy, deps=deps, job_id="job-1")
    second, _, _ = _run(tmp_path, policy=policy, deps=deps, job_id="job-1")
    assert second["status"] == "job_already_recorded"
    assert len(calls) == 1


# -- Standard -----------------------------------------------------------


def test_standard_publishes_only_through_the_native_writer(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)
    deps, calls, _ = _deps(tmp_path, root, writer=_writer(root))
    result, ledger, store = _run(tmp_path, policy=_policy(lite_enabled=False, standard_enabled=True), deps=deps)
    assert result["status"] == "standard_cycle_complete", result
    assert result["assessment"]["decision"] == "auto_publish"
    assert result["publish"]["published"] is True
    assert result["instruction_changed"] is True
    assert sd.package_digest(root) != before
    assert "verify, twice" in (root / "SKILL.md").read_text(encoding="utf-8")
    assert [call["role"] for call in calls] == ["reflector", "curator", "task"]
    assert store.proposals(job_id="job-1")[0]["decision"] == "auto_publish"


def _host_tool(root: Path):
    """A stand-in for the host skill_manage: mutations only, JSON answers."""

    calls: list[dict] = []

    def manage(**kwargs):
        calls.append(kwargs)
        target = root / kwargs["file_path"]
        if kwargs["action"] == "patch":
            text = target.read_text(encoding="utf-8")
            assert kwargs["old_string"] in text
            target.write_text(
                text.replace(kwargs["old_string"], kwargs["new_string"], 1), encoding="utf-8"
            )
        return json.dumps({"success": True})

    manage.calls = calls  # type: ignore[attr-defined]
    return manage


def test_the_real_writer_reaches_the_digest_the_gate_assessed(tmp_path):
    """The two halves must agree: the writer applies, the gate re-checks the digest."""
    from hermes_self_improvement import learning_writer as lw

    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)
    tool = _host_tool(root)
    deps, calls, _ = _deps(tmp_path, root, writer=lw.build_writer(manage=tool))
    result, _, _ = _run(tmp_path, policy=_policy(lite_enabled=False, standard_enabled=True), deps=deps)

    assert result["status"] == "standard_cycle_complete", result
    assert result["assessment"]["decision"] == "auto_publish"
    publish = result["publish"]
    assert publish["published"] is True
    assert publish["verification"] == "digest_match"
    assert publish["writer"] == "host_skill_tool"
    assert sd.package_digest(root) == publish["candidate_digest"] != before
    assert [call["action"] for call in tool.calls] == ["patch"]
    assert tool.calls[0]["name"] == "demo"


def test_a_staged_host_write_is_not_published(tmp_path):
    """An approval prompt must never be recorded as a published revision."""
    from hermes_self_improvement import learning_writer as lw

    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)
    staged = lambda **kwargs: json.dumps({"success": True, "staged": True})
    deps, _, _ = _deps(tmp_path, root, writer=lw.build_writer(manage=staged))
    result, _, _ = _run(tmp_path, policy=_policy(lite_enabled=False, standard_enabled=True), deps=deps)

    # The gate approved; the host deferred. That is a deferred decision, not a
    # broken stage, so the cycle still completes and names the writer's reason.
    assert result["status"] == "standard_cycle_complete", result
    assert result["assessment"]["decision"] == "auto_publish"
    assert result["publish"]["published"] is False
    assert result["publish"]["decision"] == "human_approval_required"
    assert "staged_for_approval" in result["publish"]["reasons"][0]
    assert result["instruction_changed"] is False
    assert sd.package_digest(root) == before


def test_standard_without_a_writer_keeps_the_incumbent(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)
    deps, _, _ = _deps(tmp_path, root)
    result, _, store = _run(tmp_path, policy=_policy(lite_enabled=False, standard_enabled=True), deps=deps)
    assert result["assessment"]["decision"] == "auto_publish"
    assert result["publish"] == {
        "published": False,
        "decision": "human_approval_required",
        "reasons": ["native_writer_unavailable"],
    }
    assert result["instruction_changed"] is False
    assert sd.package_digest(root) == before
    assert store.proposals(job_id="job-1")[0]["decision"] == "auto_publish"


@pytest.mark.parametrize(
    "overrides,decision",
    [
        ({"comparison": {**GOOD_COMPARISON, "candidate_passes": 3}}, "keep_incumbent"),
        ({"comparison": {**GOOD_COMPARISON, "critical_regressions": 1}}, "keep_incumbent"),
        ({"holdout": {**GOOD_HOLDOUT, "untouched": False}}, "human_approval_required"),
        ({"scan": {"status": "caution"}}, "human_approval_required"),
    ],
)
def test_standard_gate_blocks_publication(tmp_path, overrides, decision):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)
    deps, _, _ = _deps(tmp_path, root, writer=_writer(root), **overrides)
    result, _, _ = _run(tmp_path, policy=_policy(lite_enabled=False, standard_enabled=True), deps=deps)
    assert result["assessment"]["decision"] == decision
    assert result["instruction_changed"] is False
    assert sd.package_digest(root) == before


def test_standard_rejects_a_delta_for_another_revision(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)
    deps, _, _ = _deps(tmp_path, root, writer=_writer(root), delta_factory=lambda base: {"base_digest": "f" * 64, "ops": []})
    result, _, _ = _run(tmp_path, policy=_policy(lite_enabled=False, standard_enabled=True), deps=deps)
    assert result["status"] == "stopped_invalid_stage"
    assert sd.package_digest(root) == before


def test_standard_rejects_a_traversing_delta(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)

    def escaping(base):
        return {"base_digest": base, "ops": [{"op": "write_support_file", "path": "../escape.md", "content": "x"}]}

    deps, _, _ = _deps(tmp_path, root, writer=_writer(root), delta_factory=escaping)
    result, _, _ = _run(tmp_path, policy=_policy(lite_enabled=False, standard_enabled=True), deps=deps)
    assert result["status"] == "stopped_invalid_stage"
    assert sd.package_digest(root) == before


def test_untrusted_project_scope_refuses_an_automatic_write(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "proj"
    (project / ".git").mkdir(parents=True)
    scope = ss.resolve_scope(project, home)
    root = _package(project / ".agents" / "skills" / "demo")
    before = sd.package_digest(root)
    deps, _, _ = _deps(tmp_path, root, writer=_writer(root))
    deps["home"] = home
    result, _, _ = _run(tmp_path, policy=_policy(lite_enabled=False, standard_enabled=True), deps=deps, scope=scope)
    assert result["publish"]["published"] is False
    assert any("trusted_project_dirs" in reason for reason in result["publish"]["reasons"])
    assert sd.package_digest(root) == before


def test_standard_requires_curator_and_evaluator(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root)
    deps.pop("evaluator")
    result, _, _ = _run(tmp_path, policy=_policy(lite_enabled=False, standard_enabled=True), deps=deps)
    assert result["status"] == "backend_unavailable"
    assert result["reason"] == "evaluator"
    assert calls == []


# -- Heavy --------------------------------------------------------------


def test_heavy_search_is_bounded_and_publishes_at_most_once(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root, writer=_writer(root))
    result, ledger, store = _run(
        tmp_path,
        policy=_policy(lite_enabled=False, standard_enabled=True, heavy_enabled=True),
        deps=deps,
        manifest={"rounds": 1, "variants_per_round": 1, "max_evaluations": 1, "seed": 3},
    )
    assert result["status"] == "heavy_search_complete"
    assert result["search"]["evaluations"] == 1
    assert [call["role"] for call in calls] == [
        "reflector",
        "curator",
        "task",
        "curator",
        "task",
        "reviewer",
    ]
    assert result["review"]["verdict"] == "approve"
    assert result["publish"]["published"] is True
    published_events = [row for row in store.proposals(job_id="job-1")]
    assert len(published_events) == 1
    assert ledger.job("job-1")["ceiling"] == 20


def test_heavy_respects_the_job_ceiling(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root, writer=_writer(root))
    policy = _policy(lite_enabled=False, standard_enabled=True, heavy_enabled=True)
    policy["budget"]["heavy_job_model_calls"] = 4
    result, ledger, _ = _run(
        tmp_path,
        policy=policy,
        deps=deps,
        manifest={"rounds": 2, "variants_per_round": 2, "max_evaluations": 4, "seed": 3},
    )
    assert len(calls) <= 4
    assert result["model_calls"] <= 4
    assert ledger.job("job-1")["ceiling"] == 4


def test_manifest_digest_changes_with_the_mode_combination(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    first = lm.manifest_digest(
        pipeline="lite",
        scope_id="home:/h",
        base_digest="a" * 64,
        evidence_digest="b" * 64,
        policy=_policy(lite_enabled=True, standard_enabled=False),
    )
    second = lm.manifest_digest(
        pipeline="lite+standard",
        scope_id="home:/h",
        base_digest="a" * 64,
        evidence_digest="b" * 64,
        policy=_policy(lite_enabled=True, standard_enabled=True),
    )
    assert first != second


def test_parse_lesson_keeps_evidence_and_flags_instructions():
    parsed = lm.parse_lesson(json.dumps(LESSON))
    assert parsed["instruction_changed"] is False
    assert parsed["operational_instruction"] is False
    imperative = lm.parse_lesson(json.dumps({**LESSON, "observation": "Never split on spaces."}))
    assert imperative["operational_instruction"] is True
    with pytest.raises(lm.LearningError):
        lm.parse_lesson(json.dumps({**LESSON, "confidence": "certain"}))
    with pytest.raises(lm.LearningError):
        lm.parse_lesson(json.dumps({**LESSON, "evidence_ids": []}))
    assert lm.parse_lesson("```json\n" + json.dumps(LESSON) + "\n```")["kind"] == "lesson"


# -- M2/M4/M5 additions --------------------------------------------------


def test_deadline_stops_before_any_dispatch(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root)
    result, ledger, _ = _run(
        tmp_path,
        policy=_policy(lite_enabled=True, standard_enabled=False),
        deps=deps,
        deadline=0.0,
    )
    assert result["status"] == "deadline_exceeded"
    assert result["model_calls"] == 0
    assert calls == []
    assert ledger.job("job-1")["outcome"] == "deadline_exceeded"


def test_identical_evidence_is_not_charged_twice(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root)
    policy = _policy(lite_enabled=True, standard_enabled=False)
    first, ledger, store = _run(tmp_path, policy=policy, deps=deps, job_id="job-1")
    assert first["status"] == "lesson_saved"
    second, _, _ = _run(tmp_path, policy=policy, deps=deps, job_id="job-2", dedupe=True)
    assert second["status"] == "lesson_duplicate"
    assert second["model_calls"] == 0
    assert second["lesson_record"]["job_id"] == "job-1"
    assert [call["role"] for call in calls] == ["reflector"]
    assert ledger.job("job-2") is None


def test_reviewer_can_block_publication_without_changing_the_gate(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(
        tmp_path,
        root,
        writer=_writer(root),
        review={"verdict": "changes_required", "findings": ["the delta over-claims"]},
    )
    before = sd.package_digest(root)
    result, _, _ = _run(
        tmp_path,
        policy=_policy(lite_enabled=False, standard_enabled=True, heavy_enabled=True),
        deps=deps,
        manifest={"rounds": 1, "variants_per_round": 1, "max_evaluations": 1, "seed": 3},
    )
    assert result["review"]["verdict"] == "changes_required"
    assert result["assessment"]["decision"] == "auto_publish"
    assert result["publish"]["published"] is False
    assert result["publish"]["reasons"] == ["reviewer_changes_required"]
    assert sd.package_digest(root) == before
    assert calls[-1]["role"] == "reviewer"


def test_heavy_without_a_reviewer_is_backend_unavailable(tmp_path):
    root = _package(tmp_path / "home" / ".agents" / "skills" / "demo")
    deps, calls, _ = _deps(tmp_path, root, writer=_writer(root))
    deps.pop("reviewer")
    result, _, _ = _run(
        tmp_path,
        policy=_policy(lite_enabled=False, standard_enabled=True, heavy_enabled=True),
        deps=deps,
        manifest={"rounds": 1, "variants_per_round": 1, "max_evaluations": 1, "seed": 3},
    )
    assert result["status"] == "backend_unavailable"
    assert result["reason"] == "reviewer"
    assert calls == []
