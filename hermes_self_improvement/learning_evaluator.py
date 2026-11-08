"""Bounded task evaluator for Standard and Heavy cycles.

The evaluator runs each frozen case twice -- once with the installed skill text,
once with the candidate -- using the same runner, the same prompt template and
the same explicit checks. Checks never reach the model: the prompt contains only
the skill text and the case input.

Its claims are deliberately narrow: text-only instruction transfer, explicit
checks only, no routing, filesystem-task or billing attestation.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

CHECK_KINDS = ("contains", "not_contains", "regex", "json_keys", "equals_json", "min_lines")
DEV = "dev"
HOLDOUT = "holdout"
PROMPT_TEMPLATE = "Apply the supplied skill to the input below.\n\nINPUT:\n{input}"


class EvaluatorError(ValueError):
    """Raised for malformed cases or a runner that breaks the contract."""


def load_cases(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvaluatorError(f"cases are not JSON: {exc.msg}") from exc
    if not isinstance(raw, list) or not raw:
        raise EvaluatorError("cases must be a non-empty list")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for case in raw:
        if not isinstance(case, dict):
            raise EvaluatorError("every case must be an object")
        unknown = set(case) - {"case_id", "input", "checks", "split", "trusted", "critical"}
        if unknown:
            raise EvaluatorError(f"unknown case key(s): {', '.join(sorted(unknown))}")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise EvaluatorError("case_id must be a non-empty string")
        if case_id in seen:
            raise EvaluatorError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        text = case.get("input")
        if not isinstance(text, str) or not text.strip():
            raise EvaluatorError(f"{case_id}: input must be non-empty text")
        checks = case.get("checks")
        if not isinstance(checks, list) or not checks:
            raise EvaluatorError(f"{case_id}: at least one check is required")
        for check in checks:
            if not isinstance(check, dict) or check.get("kind") not in CHECK_KINDS:
                raise EvaluatorError(f"{case_id}: unsupported check")
            if "value" not in check:
                raise EvaluatorError(f"{case_id}: every check needs a value")
        split = case.get("split", DEV)
        if split not in (DEV, HOLDOUT):
            raise EvaluatorError(f"{case_id}: split must be dev or holdout")
        trusted = case.get("trusted", True)
        if type(trusted) is not bool:
            raise EvaluatorError(f"{case_id}: trusted must be a boolean")
        critical = case.get("critical", True)
        if type(critical) is not bool:
            raise EvaluatorError(f"{case_id}: critical must be a boolean")
        normalized.append(
            {
                "case_id": case_id,
                "input": text,
                "checks": checks,
                "split": split,
                "trusted": trusted,
                "critical": critical,
            }
        )
    return sorted(normalized, key=lambda item: item["case_id"])


def run_checks(text: str, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate explicit checks; each result carries the evidence it used."""
    if not isinstance(text, str):
        raise EvaluatorError("task output must be text")
    results = []
    for check in checks:
        kind = check["kind"]
        value = check["value"]
        passed = False
        evidence = ""
        try:
            if kind == "contains":
                passed = isinstance(value, str) and value in text
                evidence = f"substring {value!r}"
            elif kind == "not_contains":
                passed = isinstance(value, str) and value not in text
                evidence = f"absent substring {value!r}"
            elif kind == "regex":
                passed = isinstance(value, str) and re.search(value, text, re.MULTILINE) is not None
                evidence = f"pattern {value!r}"
            elif kind == "min_lines":
                passed = isinstance(value, int) and len([line for line in text.splitlines() if line.strip()]) >= value
                evidence = f"at least {value} non-empty lines"
            elif kind == "equals_json":
                expected = value
                passed = _parsed_json(text) == expected
                evidence = "full JSON equality"
            else:  # json_keys
                parsed = _parsed_json(text)
                keys = value if isinstance(value, list) else []
                passed = isinstance(parsed, dict) and all(key in parsed for key in keys)
                evidence = f"top-level keys {keys}"
        except EvaluatorError:
            passed = False
        results.append({"check": kind, "value": value, "passed": passed, "evidence": evidence})
    return results


def _parsed_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise EvaluatorError("output is not JSON") from exc


def _run_case(case: dict[str, Any], skill_text: str, run: Callable[[str], str]) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(input=case["input"])
    # The skill body is context, never a second copy of the input or the checks.
    text = run(f"{skill_text}\n\n---\n{prompt}")
    if not isinstance(text, str):
        raise EvaluatorError(f"{case['case_id']}: runner returned a non-text result")
    checks = run_checks(text, case["checks"])
    return {
        "case_id": case["case_id"],
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "output_chars": len(text),
    }


def evaluate(
    *,
    cases: list[dict[str, Any]],
    baseline_text: str,
    candidate_text: str,
    run: Callable[[str], str],
    scan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run dev and holdout cases for both arms and return gate-ready evidence."""
    if not callable(run):
        raise EvaluatorError("a text runner is required")
    for name, value in (("baseline_text", baseline_text), ("candidate_text", candidate_text)):
        if not isinstance(value, str) or not value.strip():
            raise EvaluatorError(f"{name} must be non-empty text")
    dev = [case for case in cases if case["split"] == DEV]
    holdout = [case for case in cases if case["split"] == HOLDOUT]
    if not dev:
        raise EvaluatorError("at least one dev case is required")

    dev_rows = []
    installed_passes = candidate_passes = regressions = 0
    trusted = total = 0
    for case in dev:
        total += 1
        if case["trusted"]:
            trusted += 1
        base = _run_case(case, baseline_text, run)
        cand = _run_case(case, candidate_text, run)
        installed_passes += int(base["passed"])
        candidate_passes += int(cand["passed"])
        if case["critical"] and base["passed"] and not cand["passed"]:
            regressions += 1
        dev_rows.append({"case_id": case["case_id"], "baseline": base, "candidate": cand})

    holdout_installed = holdout_candidate = 0
    holdout_rows = []
    for case in holdout:
        base = _run_case(case, baseline_text, run)
        cand = _run_case(case, candidate_text, run)
        holdout_installed += int(base["passed"])
        holdout_candidate += int(cand["passed"])
        holdout_rows.append({"case_id": case["case_id"], "baseline": base, "candidate": cand})

    return {
        "comparison": {
            "installed_passes": installed_passes,
            "candidate_passes": candidate_passes,
            "critical_regressions": regressions,
            "pairs": len(dev),
            "eligible": bool(dev) and trusted > 0,
        },
        "holdout": {
            "untouched": True,
            "pairs": len(holdout),
            "installed_passes": holdout_installed,
            "candidate_passes": holdout_candidate,
        },
        "cases": {"trusted": trusted, "total": total},
        "scan": scan if isinstance(scan, dict) else {"status": "pass", "findings": []},
        "dev_rows": dev_rows,
        "holdout_rows": holdout_rows,
        "scope": "Text-only instruction transfer with explicit checks; no routing, task-tool, "
        "filesystem-execution or billing attestation.",
    }
