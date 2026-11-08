from __future__ import annotations

import pytest

from hermes_self_improvement import skill_exposure as se

D1 = "a" * 64
D2 = "b" * 64


def _store(tmp_path):
    return se.ExposureStore(tmp_path / "exposure.sqlite3")


def test_exposure_is_not_a_success(tmp_path):
    store = _store(tmp_path)
    outcome = store.record_exposure(
        event_id="e1", skill_name="demo", skill_digest=D1, scope_id="home:/h", session_id="s1"
    )
    assert outcome["duplicate"] is False
    summary = store.summarize(skill_name="demo")
    assert summary["versions"] == [
        {"skill_digest": D1, "exposures": 1, "unbound": 1, "passed": 0, "failed": 0, "unknown": 1}
    ]
    assert "not a success" in summary["scope"]


def test_duplicate_event_is_not_counted_twice(tmp_path):
    store = _store(tmp_path)
    first = store.record_exposure(
        event_id="e1", skill_name="demo", skill_digest=D1, scope_id="home:/h", session_id="s1"
    )
    second = store.record_exposure(
        event_id="e1", skill_name="demo", skill_digest=D1, scope_id="home:/h", session_id="s1"
    )
    assert second["duplicate"] is True
    assert first["exposure_id"] == second["exposure_id"]
    assert store.summarize(skill_name="demo")["versions"][0]["exposures"] == 1


def test_same_event_for_another_version_is_refused(tmp_path):
    store = _store(tmp_path)
    store.record_exposure(event_id="e1", skill_name="demo", skill_digest=D1, scope_id="h", session_id="s")
    with pytest.raises(se.ExposureError):
        store.record_exposure(event_id="e1", skill_name="demo", skill_digest=D2, scope_id="h", session_id="s")


def test_confirmed_outcome_is_bound_and_counted(tmp_path):
    store = _store(tmp_path)
    exposure = store.record_exposure(
        event_id="e1", skill_name="demo", skill_digest=D1, scope_id="h", session_id="s"
    )
    bound = store.bind_confirmed_outcome(
        exposure["exposure_id"],
        outcome_id="o1",
        kind="task_passed",
        passed=True,
        evidence_ids=["trial-9"],
        expected_digest=D1,
    )
    assert bound["bound"] is True
    summary = store.summarize(skill_name="demo")["versions"][0]
    assert (summary["passed"], summary["unknown"]) == (1, 0)


def test_version_change_blocks_binding(tmp_path):
    store = _store(tmp_path)
    exposure = store.record_exposure(event_id="e1", skill_name="demo", skill_digest=D1, scope_id="h", session_id="s")
    with pytest.raises(se.ExposureError, match="version changed"):
        store.bind_confirmed_outcome(
            exposure["exposure_id"],
            outcome_id="o1",
            kind="task_passed",
            passed=True,
            evidence_ids=["t"],
            expected_digest=D2,
        )


def test_unknown_outcome_must_not_carry_a_verdict(tmp_path):
    store = _store(tmp_path)
    exposure = store.record_exposure(event_id="e1", skill_name="demo", skill_digest=D1, scope_id="h", session_id="s")
    unknown = store.bind_confirmed_outcome(
        exposure["exposure_id"],
        outcome_id="o1",
        kind="unknown",
        passed=None,
        evidence_ids=["t"],
        expected_digest=D1,
    )
    assert unknown["bound"] is True
    with pytest.raises(se.ExposureError):
        store.bind_confirmed_outcome(
            exposure["exposure_id"],
            outcome_id="o2",
            kind="task_failed",
            passed=None,
            evidence_ids=["t"],
            expected_digest=D1,
        )


def test_binding_needs_evidence_and_an_existing_exposure(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(se.ExposureError):
        store.bind_confirmed_outcome(
            "missing", outcome_id="o1", kind="task_passed", passed=True, evidence_ids=["t"], expected_digest=D1
        )
    exposure = store.record_exposure(event_id="e1", skill_name="demo", skill_digest=D1, scope_id="h", session_id="s")
    with pytest.raises(se.ExposureError):
        store.bind_confirmed_outcome(
            exposure["exposure_id"],
            outcome_id="o1",
            kind="task_passed",
            passed=True,
            evidence_ids=[],
            expected_digest=D1,
        )


def test_versions_are_never_merged_in_a_summary(tmp_path):
    store = _store(tmp_path)
    first = store.record_exposure(event_id="e1", skill_name="demo", skill_digest=D1, scope_id="h", session_id="s")
    second = store.record_exposure(event_id="e2", skill_name="demo", skill_digest=D2, scope_id="h", session_id="s")
    store.bind_confirmed_outcome(
        first["exposure_id"], outcome_id="o1", kind="task_passed", passed=True, evidence_ids=["t"], expected_digest=D1
    )
    store.bind_confirmed_outcome(
        second["exposure_id"], outcome_id="o2", kind="task_failed", passed=False, evidence_ids=["t"], expected_digest=D2
    )
    versions = store.summarize(skill_name="demo")["versions"]
    assert [row["skill_digest"] for row in versions] == [D1, D2]
    assert versions[0]["passed"] == 1 and versions[0]["failed"] == 0
    assert versions[1]["failed"] == 1 and versions[1]["passed"] == 0


def test_offline_replay_is_order_and_duplicate_insensitive():
    events = [
        {"event_id": "e1", "kind": "exposure", "skill_digest": D1},
        {"event_id": "e2", "kind": "exposure", "skill_digest": D1},
        {"event_id": "o1", "kind": "outcome", "exposure_event_id": "e1", "skill_digest": D1, "passed": True},
    ]
    forward = se.summarize_events(events)
    reordered = se.summarize_events(list(reversed(events)) + events[:1])
    assert forward == reordered
    assert forward["versions"][0]["exposures"] == 2
    assert forward["versions"][0]["passed"] == 1
    assert forward["versions"][0]["unknown"] == 1


def test_replay_rejects_malformed_events():
    with pytest.raises(se.ExposureError):
        se.summarize_events([{"event_id": "e1", "kind": "view", "skill_digest": D1}])
    with pytest.raises(se.ExposureError):
        se.summarize_events([{"event_id": "e1", "kind": "exposure", "skill_digest": "short"}])
    with pytest.raises(se.ExposureError):
        se.summarize_events(
            [{"event_id": "o1", "kind": "outcome", "skill_digest": D1, "passed": True}]
        )
