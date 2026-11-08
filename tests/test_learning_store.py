from __future__ import annotations

import pytest

from hermes_self_improvement.learning_store import LessonStore, StoreError

LESSON = {
    "kind": "lesson",
    "observation": "the parser dropped empty fields",
    "hypothesis": "splitting on tabs keeps trailing empties",
    "evidence_ids": ["t1"],
    "confidence": "medium",
    "scope": "project",
}


def _store(tmp_path):
    return LessonStore(tmp_path / "lessons.sqlite3")


def test_lesson_round_trip_and_scope_query(tmp_path):
    store = _store(tmp_path)
    saved = store.save_lesson(job_id="job-1", scope_id="project:/p", lesson=LESSON)
    assert saved["idempotent"] is False
    row = store.lesson("job-1")
    assert row["lesson"] == LESSON
    assert row["status"] == "observation"
    assert store.recent_lessons(scope_id="project:/p")[0]["job_id"] == "job-1"
    assert store.recent_lessons(scope_id="home:/h") == []


def test_same_job_cannot_save_two_different_lessons(tmp_path):
    store = _store(tmp_path)
    store.save_lesson(job_id="job-1", scope_id="s", lesson=LESSON)
    again = store.save_lesson(job_id="job-1", scope_id="s", lesson=LESSON)
    assert again["idempotent"] is True
    with pytest.raises(StoreError):
        store.save_lesson(job_id="job-1", scope_id="s", lesson={**LESSON, "observation": "other"})


def test_unknown_status_and_bad_limits_are_rejected(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(StoreError):
        store.save_lesson(job_id="job-1", scope_id="s", lesson=LESSON, status="approved")
    with pytest.raises(StoreError):
        store.recent_lessons(limit=0)
    with pytest.raises(StoreError):
        store.recent_lessons(limit=10_000)


def test_proposal_records_keep_the_decision(tmp_path):
    store = _store(tmp_path)
    saved = store.save_proposal(
        job_id="job-1",
        scope_id="s",
        base_digest="a" * 64,
        candidate_digest="b" * 64,
        decision="human_approval_required",
        payload={"reasons": ["quality_gain"]},
    )
    rows = store.proposals(job_id="job-1")
    assert rows[0]["proposal_id"] == saved["proposal_id"]
    assert rows[0]["decision"] == "human_approval_required"
    assert rows[0]["payload"]["reasons"] == ["quality_gain"]
    with pytest.raises(StoreError):
        store.save_proposal(
            job_id="job-1",
            scope_id="s",
            base_digest="a" * 64,
            candidate_digest="b" * 64,
            decision="approved",
            payload={},
        )


# -- M5: evidence dedupe and recurring observations -----------------------


def test_evidence_digest_finds_the_same_lesson_within_one_scope(tmp_path):
    store = _store(tmp_path)
    digest = "a" * 64
    store.save_lesson(job_id="job-1", scope_id="project:/p", lesson=LESSON, evidence_digest=digest)
    found = store.find_by_evidence(scope_id="project:/p", evidence_digest=digest)
    assert found["job_id"] == "job-1"
    assert found["lesson"]["observation"] == LESSON["observation"]
    assert store.find_by_evidence(scope_id="home:/h", evidence_digest=digest) is None
    with pytest.raises(StoreError):
        store.find_by_evidence(scope_id="project:/p", evidence_digest="short")


def test_recurring_groups_only_repeated_observations(tmp_path):
    store = _store(tmp_path)
    store.save_lesson(job_id="job-1", scope_id="project:/p", lesson=LESSON)
    store.save_lesson(
        job_id="job-2",
        scope_id="project:/p",
        lesson={**LESSON, "observation": "  The   parser dropped empty fields "},
    )
    store.save_lesson(job_id="job-3", scope_id="project:/p", lesson={**LESSON, "observation": "another defect"})

    groups = store.recurring(scope_id="project:/p")
    assert len(groups) == 1
    assert groups[0]["count"] == 2
    assert groups[0]["job_ids"] == ["job-1", "job-2"]
    assert "paired cycle" in groups[0]["scope"]

    with pytest.raises(StoreError):
        store.recurring(min_occurrences=1)


def test_find_lesson_matches_the_normalized_text(tmp_path):
    store = _store(tmp_path)
    store.save_lesson(job_id="job-1", scope_id="project:/p", lesson=LESSON)
    assert store.find_lesson(scope_id="project:/p", observation="THE PARSER   dropped empty fields")["job_id"] == "job-1"
    assert store.find_lesson(scope_id="project:/p", observation="unrelated") is None
