from __future__ import annotations

import pytest

from hermes_self_improvement import learning_search as ls

MANIFEST = {"rounds": 2, "variants_per_round": 2, "max_evaluations": 4, "seed": 7}


def test_manifest_defaults_and_caps():
    resolved = ls.validate_manifest({})
    assert resolved["rounds"] == 2
    assert resolved["variants_per_round"] == 2
    with pytest.raises(ls.SearchError):
        ls.validate_manifest({"rounds": ls.MAX_ROUNDS + 1})
    with pytest.raises(ls.SearchError):
        ls.validate_manifest({"rounds": 3, "variants_per_round": 3, "max_evaluations": 4})
    with pytest.raises(ls.SearchError):
        ls.validate_manifest({"unknown": 1})
    with pytest.raises(ls.SearchError):
        ls.validate_manifest({"rounds": True})


def test_pareto_front_drops_dominated_rows():
    rows = [
        {"delta_id": "a", "candidate_passes": 4, "critical_regressions": 0, "changed_characters": 100},
        {"delta_id": "b", "candidate_passes": 3, "critical_regressions": 0, "changed_characters": 100},
        {"delta_id": "c", "candidate_passes": 4, "critical_regressions": 0, "changed_characters": 400},
        {"delta_id": "d", "candidate_passes": 2, "critical_regressions": 1, "changed_characters": 200},
    ]
    front = ls.pareto_front(rows)
    assert [row["delta_id"] for row in front] == ["a"]


def test_pareto_front_keeps_a_cheaper_equal_quality_row():
    rows = [
        {"delta_id": "big", "candidate_passes": 4, "critical_regressions": 0, "changed_characters": 900},
        {"delta_id": "small", "candidate_passes": 4, "critical_regressions": 0, "changed_characters": 90},
    ]
    assert [row["delta_id"] for row in ls.pareto_front(rows)] == ["small"]


def test_pareto_front_keeps_cheap_but_weaker_rows():
    # Cheaper text with fewer passes is a genuine trade-off, not domination.
    rows = [
        {"delta_id": "strong", "candidate_passes": 4, "critical_regressions": 0, "changed_characters": 900},
        {"delta_id": "cheap", "candidate_passes": 3, "critical_regressions": 0, "changed_characters": 90},
    ]
    assert [row["delta_id"] for row in ls.pareto_front(rows)] == ["strong", "cheap"]


def test_pareto_front_requires_ids_and_integers():
    with pytest.raises(ls.SearchError):
        ls.pareto_front([{"candidate_passes": 1}])
    with pytest.raises(ls.SearchError):
        ls.pareto_front([{"delta_id": "a", "candidate_passes": "1"}])
    with pytest.raises(ls.SearchError):
        ls.pareto_front({"delta_id": "a"})


def test_parent_selection_is_seeded_and_reproducible():
    front = [
        {"delta_id": f"d{index}", "candidate_passes": 5 - index, "critical_regressions": 0, "changed_characters": 10}
        for index in range(5)
    ]
    first = ls.select_parents(front, seed=3, count=2)
    second = ls.select_parents(front, seed=3, count=2)
    assert [row["delta_id"] for row in first] == [row["delta_id"] for row in second]
    assert first[0]["delta_id"] == "d0"
    assert first is not front


def test_flat_round_detects_no_improvement():
    rows = [{"delta_id": "a", "candidate_passes": 3}]
    assert ls.flat_round(rows, 3) is True
    assert ls.flat_round(rows, 2) is False


def _runner(evaluations, budget=10, manifest=None):
    calls: list[dict] = []
    resolved = manifest or MANIFEST

    def propose(round_index, parents):
        calls.append({"round": round_index, "parents": [p["delta_id"] for p in parents]})
        return [
            {"delta_id": f"r{round_index}v{index}", "delta": {"changed_characters": 10 + index}}
            for index in range(resolved["variants_per_round"])
        ]

    def evaluate(variant):
        return evaluations.pop(0)

    return ls.run_search(
        manifest=resolved,
        incumbent_passes=2,
        base_digest="a" * 64,
        propose=propose,
        evaluate=evaluate,
        budget_left=lambda: budget,
    ), calls


def test_search_stops_when_budget_is_gone():
    result, calls = _runner(evaluations=[], budget=0)
    assert result["stopped_reason"] == "budget_exhausted"
    assert result["rows"] == []
    assert calls == []


def test_search_records_rows_and_reports_improvement():
    good = {"candidate_passes": 4, "critical_regressions": 0}
    flat = {"candidate_passes": 2, "critical_regressions": 0}
    result, _ = _runner(evaluations=[good, flat, flat, flat], budget=10)
    assert result["evaluations"] == 4
    assert result["improved"] is True
    assert result["best"]["delta_id"] == "r0v0"
    assert result["best"]["base_digest"] == "a" * 64


def test_search_stops_after_two_flat_rounds():
    flat = {"candidate_passes": 2, "critical_regressions": 0}
    manifest = {"rounds": 3, "variants_per_round": 1, "max_evaluations": 3, "seed": 1}
    result, calls = _runner(evaluations=[flat, flat], manifest=manifest, budget=10)
    assert result["stopped_reason"] == "no_improvement"
    assert result["improved"] is False
    assert len(calls) == 2


def test_search_skips_oversized_variants():
    manifest = {"rounds": 1, "variants_per_round": 2, "max_evaluations": 2, "max_changed_chars": 10, "seed": 1}
    result, _ = _runner(
        evaluations=[{"candidate_passes": 5, "critical_regressions": 0}], manifest=manifest, budget=10
    )
    assert result["evaluations"] == 1
    assert result["rows"][0]["changed_characters"] == 10
