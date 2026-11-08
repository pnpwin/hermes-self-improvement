from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hermes_self_improvement import learning_publish as pub
from hermes_self_improvement import skill_delta as sd

BASE = "a" * 64
CAND = "b" * 64
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


def assess(**overrides):
    kwargs = {
        "base_digest": BASE,
        "candidate_digest": CAND,
        "comparison": dict(GOOD_COMPARISON),
        "holdout": dict(GOOD_HOLDOUT),
        "cases": dict(GOOD_CASES),
        "scan": dict(CLEAN_SCAN),
    }
    kwargs.update(overrides)
    return pub.assess_candidate(**kwargs)


def test_all_gates_clear_is_auto_publish():
    result = assess()
    assert result["decision"] == "auto_publish"
    assert result["reasons"] == []
    assert all(result["gates"].values())


def test_identical_digest_is_no_change_and_keeps_incumbent():
    result = assess(candidate_digest=BASE)
    assert result["decision"] == "keep_incumbent"
    assert result["reasons"] == ["no_change"]


def test_critical_regression_blocks_publication():
    result = assess(comparison={**GOOD_COMPARISON, "critical_regressions": 1})
    assert result["decision"] == "keep_incumbent"
    assert "no_critical_regression" in result["reasons"]


def test_no_quality_gain_keeps_the_incumbent():
    result = assess(comparison={**GOOD_COMPARISON, "candidate_passes": 3})
    assert result["decision"] == "keep_incumbent"
    assert result["reasons"] == ["quality_gain"]


def test_untouched_holdout_is_required_for_automation():
    result = assess(holdout={**GOOD_HOLDOUT, "untouched": False})
    assert result["decision"] == "human_approval_required"
    assert "holdout_untouched" in result["reasons"]
    missing_pairs = assess(holdout={"untouched": True, "pairs": 0, "installed_passes": 0, "candidate_passes": 0})
    assert "holdout_untouched" in missing_pairs["reasons"]


def test_holdout_regression_requires_a_human():
    result = assess(holdout={**GOOD_HOLDOUT, "candidate_passes": 1})
    assert result["decision"] == "human_approval_required"
    assert "holdout_not_worse" in result["reasons"]


def test_no_trusted_cases_and_unclean_scanner_block_automation():
    assert "trusted_cases_present" in assess(cases={"trusted": 0, "total": 5})["reasons"]
    assert "scanner_clean" in assess(scan={"status": "caution"})["reasons"]
    assert "scanner_clean" in assess(scan={"status": "fail"})["reasons"]


def test_ineligible_comparison_is_not_a_quality_claim():
    result = assess(comparison={**GOOD_COMPARISON, "eligible": False})
    assert result["decision"] == "human_approval_required"
    assert "paired_comparison_eligible" in result["reasons"]


def test_malformed_evidence_is_rejected_before_a_decision():
    with pytest.raises(pub.PublishError):
        assess(comparison={**GOOD_COMPARISON, "candidate_passes": "4"})
    with pytest.raises(pub.PublishError):
        assess(comparison={**GOOD_COMPARISON, "eligible": "true"})
    with pytest.raises(pub.PublishError):
        assess(holdout={**GOOD_HOLDOUT, "untouched": "yes"})
    with pytest.raises(pub.PublishError):
        assess(scan={"status": "unknown"})
    with pytest.raises(pub.PublishError):
        assess(base_digest="short")


def _package(root: Path) -> Path:
    (root / "references").mkdir(parents=True)
    (root / "SKILL.md").write_text("# Skill\n\nAlways verify.\n", encoding="utf-8")
    (root / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
    return root


def test_publish_requires_auto_publish_decision(tmp_path):
    with pytest.raises(pub.PublishError):
        pub.publish_candidate(
            decision={"decision": "human_approval_required"},
            delta={"ops": []},
            staging_root=tmp_path,
            target_root=tmp_path,
            native_writer=lambda **_: {"published": True},
            expected_base_digest=BASE,
            expected_candidate_digest=CAND,
        )


def test_publish_requires_a_native_writer(tmp_path):
    root = _package(tmp_path / "target")
    staging = tmp_path / "staging"
    shutil.copytree(root, staging)
    with pytest.raises(pub.PublishError, match="native_writer_unavailable"):
        pub.publish_candidate(
            decision={"decision": "auto_publish"},
            delta={"ops": []},
            staging_root=staging,
            target_root=root,
            native_writer=None,
            expected_base_digest=sd.package_digest(root),
            expected_candidate_digest=sd.package_digest(staging),
        )


def test_publish_refuses_a_drifted_installed_package(tmp_path):
    root = _package(tmp_path / "target")
    staging = tmp_path / "staging"
    shutil.copytree(root, staging)
    (staging / "SKILL.md").write_text("# Skill\n\nAlways verify twice.\n", encoding="utf-8")
    (root / "SKILL.md").write_text("# Skill\n\nDrifted.\n", encoding="utf-8")
    with pytest.raises(pub.PublishError, match="drifted"):
        pub.publish_candidate(
            decision={"decision": "auto_publish"},
            delta={"ops": []},
            staging_root=staging,
            target_root=root,
            native_writer=lambda **_: {"published": True},
            expected_base_digest="c" * 64,
            expected_candidate_digest=sd.package_digest(staging),
        )


def test_publish_verifies_the_written_revision(tmp_path):
    root = _package(tmp_path / "target")
    staging = tmp_path / "staging"
    shutil.copytree(root, staging)
    (staging / "SKILL.md").write_text("# Skill\n\nAlways verify twice.\n", encoding="utf-8")
    base = sd.package_digest(root)
    candidate = sd.package_digest(staging)

    def honest_writer(**kwargs):
        shutil.copy2(staging / "SKILL.md", root / "SKILL.md")
        return {"published": True, "writer": "test-native"}

    result = pub.publish_candidate(
        decision={"decision": "auto_publish"},
        delta={"ops": []},
        staging_root=staging,
        target_root=root,
        native_writer=honest_writer,
        expected_base_digest=base,
        expected_candidate_digest=candidate,
    )
    assert result["published"] is True
    assert result["candidate_digest"] == candidate


def test_writer_that_does_not_write_is_refused(tmp_path):
    root = _package(tmp_path / "target")
    staging = tmp_path / "staging"
    shutil.copytree(root, staging)
    (staging / "SKILL.md").write_text("# Skill\n\nAlways verify twice.\n", encoding="utf-8")

    with pytest.raises(pub.PublishError, match="does not match"):
        pub.publish_candidate(
            decision={"decision": "auto_publish"},
            delta={"ops": []},
            staging_root=staging,
            target_root=root,
            native_writer=lambda **_: {"published": True},
            expected_base_digest=sd.package_digest(root),
            expected_candidate_digest=sd.package_digest(staging),
        )
