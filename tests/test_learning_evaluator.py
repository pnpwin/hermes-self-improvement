from __future__ import annotations

import json

import pytest

from hermes_self_improvement import learning_evaluator as le

CASES = [
    {"case_id": "D1", "input": "a\tb\tc", "checks": [{"kind": "json_keys", "value": ["fields"]}]},
    {"case_id": "D2", "input": "no tabs", "checks": [{"kind": "contains", "value": "error"}]},
    {"case_id": "H1", "input": "x", "split": "holdout", "checks": [{"kind": "contains", "value": "ok"}]},
]


def test_cases_are_validated_and_sorted():
    loaded = le.load_cases(CASES)
    assert [case["case_id"] for case in loaded] == ["D1", "D2", "H1"]
    assert loaded[0]["split"] == "dev"
    with pytest.raises(le.EvaluatorError):
        le.load_cases([])
    with pytest.raises(le.EvaluatorError):
        le.load_cases([{**CASES[0], "case_id": "D1"}, CASES[0]])
    with pytest.raises(le.EvaluatorError):
        le.load_cases([{**CASES[0], "checks": []}])
    with pytest.raises(le.EvaluatorError):
        le.load_cases([{**CASES[0], "checks": [{"kind": "vibes", "value": 1}]}])
    with pytest.raises(le.EvaluatorError):
        le.load_cases([{**CASES[0], "split": "train"}])
    with pytest.raises(le.EvaluatorError):
        le.load_cases([{**CASES[0], "trusted": "yes"}])
    with pytest.raises(le.EvaluatorError):
        le.load_cases([{**CASES[0], "extra": 1}])


def test_checks_cover_each_declared_kind():
    assert le.run_checks('{"fields": 3}', [{"kind": "json_keys", "value": ["fields"]}])[0]["passed"] is True
    assert le.run_checks("total 3", [{"kind": "contains", "value": "total"}])[0]["passed"] is True
    assert le.run_checks("nothing", [{"kind": "not_contains", "value": "total"}])[0]["passed"] is True
    assert le.run_checks("id=42", [{"kind": "regex", "value": r"id=\d+"}])[0]["passed"] is True
    assert le.run_checks("a\nb\n", [{"kind": "min_lines", "value": 2}])[0]["passed"] is True
    assert le.run_checks('[1, 2]', [{"kind": "equals_json", "value": [1, 2]}])[0]["passed"] is True
    assert le.run_checks("not json", [{"kind": "json_keys", "value": ["a"]}])[0]["passed"] is False


def test_prompt_never_contains_expected_values():
    seen: list[str] = []

    def run(prompt: str) -> str:
        seen.append(prompt)
        return '{"fields": 3}'

    le.evaluate(
        cases=le.load_cases([{**CASES[0], "checks": [{"kind": "json_keys", "value": ["secret_key"]}]}]),
        baseline_text="BASELINE SKILL",
        candidate_text="CANDIDATE SKILL",
        run=run,
    )
    assert all("secret_key" not in prompt for prompt in seen)
    assert all("D1" not in prompt for prompt in seen)


def test_candidate_gain_and_holdout_are_measured_separately():
    def run(prompt: str) -> str:
        if "CANDIDATE" in prompt:
            return json.dumps({"fields": 3}) if "a\tb\tc" in prompt else "an error"
        return "nothing useful"

    result = le.evaluate(
        cases=le.load_cases(CASES), baseline_text="BASELINE", candidate_text="CANDIDATE", run=run
    )
    assert result["comparison"] == {
        "installed_passes": 0,
        "candidate_passes": 2,
        "critical_regressions": 0,
        "pairs": 2,
        "eligible": True,
    }
    assert result["holdout"]["candidate_passes"] == 0
    assert result["cases"] == {"trusted": 2, "total": 2}


def test_regression_is_counted_only_for_critical_cases():
    def run(prompt: str) -> str:
        return "nothing" if "CANDIDATE" in prompt else "ok error"

    critical = le.evaluate(
        cases=le.load_cases([{**CASES[1], "critical": True}]),
        baseline_text="BASELINE",
        candidate_text="CANDIDATE",
        run=run,
    )
    assert critical["comparison"]["critical_regressions"] == 1
    soft = le.evaluate(
        cases=le.load_cases([{**CASES[1], "critical": False}]),
        baseline_text="BASELINE",
        candidate_text="CANDIDATE",
        run=run,
    )
    assert soft["comparison"]["critical_regressions"] == 0


def test_untrusted_cases_do_not_make_a_comparison_eligible():
    result = le.evaluate(
        cases=le.load_cases([{**CASES[1], "trusted": False}]),
        baseline_text="B",
        candidate_text="C",
        run=lambda prompt: "error",
    )
    assert result["comparison"]["eligible"] is False
    assert result["cases"] == {"trusted": 0, "total": 1}


def test_holdout_is_required_to_isolate_its_cases():
    result = le.evaluate(
        cases=le.load_cases(CASES), baseline_text="B", candidate_text="C", run=lambda prompt: "ok error"
    )
    assert result["holdout"]["pairs"] == 1
    assert [row["case_id"] for row in result["holdout_rows"]] == ["H1"]
    assert [row["case_id"] for row in result["dev_rows"]] == ["D1", "D2"]


def test_malformed_runner_result_is_refused():
    with pytest.raises(le.EvaluatorError):
        le.evaluate(cases=le.load_cases(CASES), baseline_text="B", candidate_text="C", run=lambda prompt: 1)
    with pytest.raises(le.EvaluatorError):
        le.evaluate(cases=le.load_cases(CASES), baseline_text="", candidate_text="C", run=lambda prompt: "x")
    with pytest.raises(le.EvaluatorError):
        le.evaluate(
            cases=le.load_cases([{**CASES[1], "split": "holdout"}]),
            baseline_text="B",
            candidate_text="C",
            run=lambda prompt: "x",
        )


def test_scan_findings_are_passed_through_unchanged():
    result = le.evaluate(
        cases=le.load_cases(CASES),
        baseline_text="B",
        candidate_text="C",
        run=lambda prompt: "ok error",
        scan={"status": "caution", "findings": ["context_exfil"]},
    )
    assert result["scan"]["status"] == "caution"
    assert result["scan"]["findings"] == ["context_exfil"]
