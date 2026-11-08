"""Learning-mode pipelines: Lite, Standard and Heavy.

One authoring skill, one owner of automatic changes. Every model call is
reserved against the profile's local ledger *before* dispatch, every candidate
revision is a bounded typed delta, and nothing is written into a skill package
unless the promotion gate and the native writer both agree.

The pipelines take injected callables (``reflector``, ``curator``, ``evaluator``,
``staging``, ``native_writer``) so the whole flow can be replayed offline and
tested without a model or a mutation.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from . import learning_policy as lp
from . import learning_publish as pub
from . import learning_search as search
from . import skill_delta as sd
from . import skill_scope as scope_mod

MAX_PROMPT_CHARS = 12_000
MAX_EVIDENCE_CHARS = 6_000

LESSON_KEYS = ("observation", "hypothesis", "evidence_ids", "confidence", "scope")
CONFIDENCE = ("low", "medium", "high")

REFLECT_PROMPT = (
    "You review one finished skill trial and extract a single operational lesson.\n"
    "Reply with JSON only, exactly these keys:\n"
    '{"observation": str, "hypothesis": str, "evidence_ids": [str], '
    '"confidence": "low"|"medium"|"high", "scope": "project"|"home"}\n'
    "Do not rewrite the skill. Do not give an instruction unless a confirmed outcome "
    "supports it; describe what was observed and what would falsify the hypothesis.\n"
)

CURATE_PROMPT = (
    "You revise a skill package with the smallest localized edit that addresses the lesson.\n"
    "Reply with JSON only, exactly these keys:\n"
    '{"base_digest": str, "ops": [op, ...]}\n'
    "Allowed ops: patch_text {op, path, old_string, new_string, replace_all}, "
    "write_support_file {op, path, content}, "
    "remove_support_file {op, path, expected_sha256}.\n"
    f"Limits: at most {sd.MAX_OPS} ops, {sd.MAX_TEXT_FILES} files, "
    f"{sd.MAX_CHANGED_CHARS} changed characters. Never rewrite the whole file, "
    "never touch files outside SKILL.md and references/*.md, never add secrets, "
    "and never weaken approval or safety rules.\n"
)

_INSTRUCTION = re.compile(
    r"(?im)^\s*(always|never|must|do not|don't|use |run |avoid )"
)

REVIEW_PROMPT = (
    "You review one bounded skill revision that already passed or failed an evidence gate.\n"
    "Reply with JSON only, exactly these keys:\n"
    '{"verdict": "approve"|"changes_required", "findings": [str]}\n'
    "Assess only the supplied delta, the recorded evidence and the gate decision. You cannot "
    "approve publication; you may only ask for changes. Do not invent test results.\n"
)


class BudgetExhausted(RuntimeError):
    """Raised when the ledger refuses a reservation; carries the exact verdict."""

    def __init__(self, verdict: dict[str, Any]):
        super().__init__(verdict.get("reason", "budget_denied"))
        self.verdict = verdict


class LearningError(RuntimeError):
    """Raised when a pipeline stage produced something outside its contract."""


class DeadlineExceeded(RuntimeError):
    """Raised when the job's wall-clock budget is exhausted before a dispatch."""


def _digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def manifest_digest(
    *,
    pipeline: str,
    scope_id: str,
    base_digest: str,
    evidence_digest: str,
    policy: dict[str, Any],
) -> str:
    return _digest(
        {
            "pipeline": pipeline,
            "scope_id": scope_id,
            "base_digest": base_digest,
            "evidence_digest": evidence_digest,
            "modes": policy["modes"],
            "budget": policy["budget"],
        }
    )


def _clip(text: str, limit: int) -> str:
    if not isinstance(text, str):
        raise LearningError("prompt material must be text")
    return text if len(text) <= limit else text[: limit - 20] + "\n…[truncated]"


def _json_object(text: str, *, stage: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise LearningError(f"{stage}: empty model response")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LearningError(f"{stage}: response is not JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise LearningError(f"{stage}: response must be a JSON object")
    return data


def parse_lesson(text: str) -> dict[str, Any]:
    data = _json_object(text, stage="reflector")
    if set(data) != set(LESSON_KEYS):
        raise LearningError("reflector response must contain exactly the lesson keys")
    for key in ("observation", "hypothesis"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise LearningError(f"lesson.{key} must be non-empty text")
    if data["confidence"] not in CONFIDENCE:
        raise LearningError("lesson.confidence must be low, medium or high")
    if data["scope"] not in ("project", "home"):
        raise LearningError("lesson.scope must be project or home")
    ids = data["evidence_ids"]
    if not isinstance(ids, list) or not ids or not all(isinstance(item, str) and item for item in ids):
        raise LearningError("lesson.evidence_ids must be a non-empty list of ids")
    instruction_like = bool(_INSTRUCTION.search(data["observation"]))
    return {
        "kind": "lesson",
        "observation": data["observation"].strip(),
        "hypothesis": data["hypothesis"].strip(),
        "evidence_ids": ids,
        "confidence": data["confidence"],
        "scope": data["scope"],
        "operational_instruction": instruction_like,
        "instruction_changed": False,
        "note": (
            "Observation only: no confirmed outcome yet, so the lesson stays out of "
            "operational instructions until a Standard cycle validates it."
            if instruction_like
            else "Observation saved; no instruction change was made."
        ),
    }


def _invoke(
    *,
    ledger,
    policy: dict[str, Any],
    job_id: str,
    index: int,
    role: str,
    day: str,
    fn: Callable[..., dict[str, Any]],
    payload: dict[str, Any],
    lite: bool,
    deadline: float | None = None,
) -> dict[str, Any]:
    route = policy["routes"].get(role)
    if route is None:
        raise LearningError(f"{role}: no explicit route is configured")
    if deadline is not None and time.monotonic() > deadline:
        raise DeadlineExceeded(f"{role}: job deadline exceeded before dispatch")
    invocation_id = f"{job_id}:{index:02d}:{role}"
    verdict = ledger.reserve_call(
        job_id,
        invocation_id,
        role=role,
        route=route,
        day=day,
        daily_cap=policy["budget"]["daily_model_calls"],
        lite_daily_cap=policy["budget"]["lite_daily_model_calls"],
        lite=lite,
        max_concurrent=policy["budget"].get("max_concurrent_model_calls"),
    )
    if not verdict.get("admitted"):
        raise BudgetExhausted(verdict)
    try:
        result = fn(payload, route)
    except Exception:
        ledger.finish_call(invocation_id, status="failed", usage={})
        raise
    if not isinstance(result, dict):
        ledger.finish_call(invocation_id, status="failed", usage={})
        raise LearningError(f"{role}: backend returned a non-object result")
    ledger.finish_call(invocation_id, status="completed", usage=result.get("usage") or {})
    return result


def _text(result: dict[str, Any], *, stage: str) -> str:
    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        raise LearningError(f"{stage}: backend returned no text")
    return text


def _reflect(
    *,
    deps: dict[str, Any],
    ledger,
    policy: dict[str, Any],
    job_id: str,
    day: str,
    evidence: dict[str, Any],
    lite: bool,
    index: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = REFLECT_PROMPT + "\nEVIDENCE:\n" + _clip(json.dumps(evidence, ensure_ascii=False, indent=2), MAX_EVIDENCE_CHARS)
    result = _invoke(
        ledger=ledger,
        policy=policy,
        job_id=job_id,
        index=index,
        role="reflector",
        day=day,
        fn=deps["reflector"],
        payload={"prompt": prompt, "evidence": evidence},
        lite=lite,
        deadline=deps.get("deadline"),
    )
    lesson = parse_lesson(_text(result, stage="reflector"))
    return lesson, result


def _curate(
    *,
    deps: dict[str, Any],
    ledger,
    policy: dict[str, Any],
    job_id: str,
    day: str,
    evidence: dict[str, Any],
    lesson: dict[str, Any],
    base_digest: str,
    index: int,
) -> dict[str, Any]:
    prompt = (
        CURATE_PROMPT
        + f"\nBASE_DIGEST: {base_digest}\nLESSON:\n"
        + _clip(json.dumps(lesson, ensure_ascii=False, indent=2), MAX_EVIDENCE_CHARS)
        + "\nEVIDENCE:\n"
        + _clip(json.dumps(evidence, ensure_ascii=False, indent=2), MAX_EVIDENCE_CHARS)
    )
    result = _invoke(
        ledger=ledger,
        policy=policy,
        job_id=job_id,
        index=index,
        role="curator",
        day=day,
        fn=deps["curator"],
        payload={"prompt": prompt, "lesson": lesson, "base_digest": base_digest},
        lite=False,
        deadline=deps.get("deadline"),
    )
    data = _json_object(_text(result, stage="curator"), stage="curator")
    return sd.validate_delta(data, base_digest=base_digest, package_root=deps["package_root"])


def _stage_and_apply(deps: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    base_root = Path(deps["package_root"]).resolve()
    staging = Path(deps["staging"](base_root)).resolve()
    if staging == base_root:
        raise LearningError("staging must be a copy, never the installed package")
    applied = sd.apply_delta(delta, package_root=staging)
    return {"staging_root": str(staging), "candidate_digest": applied["digest"], "changed_paths": applied["changed_paths"]}


def _evaluate(
    *,
    deps: dict[str, Any],
    ledger,
    policy: dict[str, Any],
    job_id: str,
    day: str,
    evidence: dict[str, Any],
    staged: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    result = _invoke(
        ledger=ledger,
        policy=policy,
        job_id=job_id,
        index=index,
        role="task",
        day=day,
        fn=deps["evaluator"],
        payload={
            "candidate_root": staged["staging_root"],
            "base_root": str(Path(deps["package_root"]).resolve()),
            "candidate_digest": staged["candidate_digest"],
            "evidence": evidence,
        },
        lite=False,
        deadline=deps.get("deadline"),
    )
    for key in ("comparison", "holdout", "cases", "scan"):
        if not isinstance(result.get(key), dict):
            raise LearningError(f"evaluator must return a {key} object")
    return result


def _finish_job(ledger, job_id: str, outcome: str) -> None:
    try:
        ledger.close_job(job_id, outcome=outcome)
    except Exception:  # pragma: no cover - closing must never mask the real result
        pass


def _prelude(
    *,
    policy: dict[str, Any],
    ledger,
    store,
    deps: dict[str, Any],
    scope: dict[str, Any],
    evidence: dict[str, Any],
    job_id: str,
    day: str,
    lite: bool,
) -> dict[str, Any]:
    lesson, reflection = _reflect(
        deps=deps, ledger=ledger, policy=policy, job_id=job_id, day=day, evidence=evidence, lite=lite
    )
    saved = store.save_lesson(
        job_id=job_id,
        scope_id=scope_mod.scope_id(scope),
        lesson=lesson,
        status="observation",
        evidence_digest=deps.get("evidence_digest"),
    )
    return {"lesson": lesson, "lesson_record": saved, "reflection_usage": reflection.get("usage") or {}}


def _candidate_and_assessment(
    *,
    policy: dict[str, Any],
    ledger,
    deps: dict[str, Any],
    scope: dict[str, Any],
    evidence: dict[str, Any],
    lesson: dict[str, Any],
    job_id: str,
    day: str,
    base_digest: str,
    index: int,
) -> dict[str, Any]:
    delta = _curate(
        deps=deps,
        ledger=ledger,
        policy=policy,
        job_id=job_id,
        day=day,
        evidence=evidence,
        lesson=lesson,
        base_digest=base_digest,
        index=index,
    )
    staged = _stage_and_apply(deps, delta)
    evaluation = _evaluate(
        deps=deps,
        ledger=ledger,
        policy=policy,
        job_id=job_id,
        day=day,
        evidence=evidence,
        staged=staged,
        index=index + 1,
    )
    assessment = pub.assess_candidate(
        base_digest=base_digest,
        candidate_digest=staged["candidate_digest"],
        comparison=evaluation["comparison"],
        holdout=evaluation["holdout"],
        cases=evaluation["cases"],
        scan=evaluation["scan"],
    )
    return {
        "delta": delta,
        "staged": staged,
        "evaluation": evaluation,
        "assessment": assessment,
    }


def _publish_if_allowed(
    *,
    deps: dict[str, Any],
    scope: dict[str, Any],
    job_id: str,
    base_digest: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    assessment = candidate["assessment"]
    if assessment["decision"] != "auto_publish":
        return {"published": False, "decision": assessment["decision"], "reasons": assessment["reasons"]}
    writer = deps.get("native_writer")
    if not callable(writer):
        return {
            "published": False,
            "decision": "human_approval_required",
            "reasons": ["native_writer_unavailable"],
        }
    target = Path(deps["package_root"]).resolve()
    try:
        scope_mod.assert_target(scope, target, home=deps.get("home"))
    except scope_mod.ScopeError as exc:
        return {"published": False, "decision": "human_approval_required", "reasons": [str(exc)]}
    try:
        result = pub.publish_candidate(
            decision=assessment,
            delta=candidate["delta"],
            staging_root=candidate["staged"]["staging_root"],
            target_root=target,
            native_writer=writer,
            expected_base_digest=base_digest,
            expected_candidate_digest=candidate["staged"]["candidate_digest"],
        )
    except pub.PublishDeferred as exc:
        # The host declined or staged the write. Nothing is wrong with the
        # candidate, so this is a deferred decision, not an invalid stage.
        return {"published": False, "decision": "human_approval_required", "reasons": [str(exc)]}
    return {"published": True, **result}


def _review(
    *,
    policy: dict[str, Any],
    ledger,
    deps: dict[str, Any],
    job_id: str,
    day: str,
    lesson: dict[str, Any],
    delta: dict[str, Any],
    assessment: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """One bounded reviewer pass: evidence only, never an approval."""
    prompt = (
        REVIEW_PROMPT
        + "\nDELTA:\n"
        + json.dumps(sd.summarize(delta), ensure_ascii=False, sort_keys=True)[:4000]
        + "\nASSESSMENT:\n"
        + json.dumps(assessment, ensure_ascii=False, sort_keys=True)[:2000]
        + "\nLESSON:\n"
        + json.dumps(lesson, ensure_ascii=False, sort_keys=True)[:2000]
    )
    result = _invoke(
        ledger=ledger,
        policy=policy,
        job_id=job_id,
        day=day,
        role="reviewer",
        index=index,
        fn=deps["reviewer"],
        payload={"prompt": prompt},
        lite=False,
        deadline=deps.get("deadline"),
    )
    data = _json_object(_text(result, stage="reviewer"), stage="reviewer")
    verdict = data.get("verdict")
    if verdict not in ("approve", "changes_required"):
        raise LearningError("reviewer verdict must be approve or changes_required")
    findings = data.get("findings") or []
    if not isinstance(findings, list) or any(not isinstance(item, str) for item in findings):
        raise LearningError("reviewer findings must be a list of strings")
    return {
        "verdict": verdict,
        "findings": findings,
        "route": policy["routes"]["reviewer"],
        "scope": "same-provider advisory review; not model independence and not an approval",
    }


def run_lite(
    *,
    policy: dict[str, Any],
    ledger,
    store,
    deps: dict[str, Any],
    scope: dict[str, Any],
    evidence: dict[str, Any],
    job_id: str,
    day: str,
) -> dict[str, Any]:
    prelude = _prelude(
        policy=policy,
        ledger=ledger,
        store=store,
        deps=deps,
        scope=scope,
        evidence=evidence,
        job_id=job_id,
        day=day,
        lite=True,
    )
    _finish_job(ledger, job_id, "lesson_saved")
    return {
        "status": "lesson_saved",
        "mode": "lite",
        "job_id": job_id,
        "lesson": prelude["lesson"],
        "lesson_record": prelude["lesson_record"],
        "instruction_changed": False,
        "package_changed": False,
        "usage": prelude["reflection_usage"],
    }


def run_standard(
    *,
    policy: dict[str, Any],
    ledger,
    store,
    deps: dict[str, Any],
    scope: dict[str, Any],
    evidence: dict[str, Any],
    job_id: str,
    day: str,
    base_digest: str,
) -> dict[str, Any]:
    prelude = _prelude(
        policy=policy,
        ledger=ledger,
        store=store,
        deps=deps,
        scope=scope,
        evidence=evidence,
        job_id=job_id,
        day=day,
        lite=False,
    )
    candidate = _candidate_and_assessment(
        policy=policy,
        ledger=ledger,
        deps=deps,
        scope=scope,
        evidence=evidence,
        lesson=prelude["lesson"],
        job_id=job_id,
        day=day,
        base_digest=base_digest,
        index=2,
    )
    store.save_proposal(
        job_id=job_id,
        scope_id=scope_mod.scope_id(scope),
        base_digest=base_digest,
        candidate_digest=candidate["staged"]["candidate_digest"],
        decision=candidate["assessment"]["decision"],
        payload={
            "delta": sd.summarize(candidate["delta"]),
            "assessment": candidate["assessment"],
        },
    )
    published = _publish_if_allowed(
        deps=deps,
        scope=scope,
        job_id=job_id,
        base_digest=base_digest,
        candidate=candidate,
    )
    outcome = "published" if published.get("published") else published.get("decision", "keep_incumbent")
    _finish_job(ledger, job_id, outcome)
    return {
        "status": "standard_cycle_complete",
        "mode": "standard",
        "job_id": job_id,
        "lesson": prelude["lesson"],
        "lesson_record": prelude["lesson_record"],
        "delta": sd.summarize(candidate["delta"]),
        "candidate_digest": candidate["staged"]["candidate_digest"],
        "assessment": candidate["assessment"],
        "publish": published,
        "instruction_changed": bool(published.get("published")),
    }


def run_heavy(
    *,
    policy: dict[str, Any],
    ledger,
    store,
    deps: dict[str, Any],
    scope: dict[str, Any],
    evidence: dict[str, Any],
    job_id: str,
    day: str,
    base_digest: str,
    search_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prelude = _prelude(
        policy=policy,
        ledger=ledger,
        store=store,
        deps=deps,
        scope=scope,
        evidence=evidence,
        job_id=job_id,
        day=day,
        lite=False,
    )
    incumbent = _candidate_and_assessment(
        policy=policy,
        ledger=ledger,
        deps=deps,
        scope=scope,
        evidence=evidence,
        lesson=prelude["lesson"],
        job_id=job_id,
        day=day,
        base_digest=base_digest,
        index=2,
    )
    manifest = search.validate_manifest(search_manifest or {})
    counter = {"index": 4}

    def propose(round_index: int, parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        variants: list[dict[str, Any]] = []
        for variant_index in range(manifest["variants_per_round"]):
            delta = _curate(
                deps=deps,
                ledger=ledger,
                policy=policy,
                job_id=job_id,
                day=day,
                evidence={**evidence, "round": round_index, "variant": variant_index, "parents": [p["delta_id"] for p in parents]},
                lesson=prelude["lesson"],
                base_digest=base_digest,
                index=counter["index"],
            )
            counter["index"] += 1
            variants.append({"delta_id": f"{job_id}:r{round_index}:v{variant_index}", "delta": delta})
        return variants

    def evaluate(variant: dict[str, Any]) -> dict[str, Any]:
        staged = _stage_and_apply(deps, variant["delta"])
        evaluation = _evaluate(
            deps=deps,
            ledger=ledger,
            policy=policy,
            job_id=job_id,
            day=day,
            evidence=evidence,
            staged=staged,
            index=counter["index"],
        )
        counter["index"] += 1
        return {
            "candidate_passes": int(evaluation["comparison"].get("candidate_passes") or 0),
            "critical_regressions": int(evaluation["comparison"].get("critical_regressions") or 0),
            "staged": staged,
            "evaluation": evaluation,
        }

    def budget_left() -> int:
        job = ledger.job(job_id) or {}
        used = len(ledger.job_invocations(job_id))
        return max(0, int(job.get("ceiling") or 0) - used) - 1  # keep one slot for the reviewer

    result = search.run_search(
        manifest=manifest,
        incumbent_passes=int(incumbent["evaluation"]["comparison"].get("candidate_passes") or 0),
        base_digest=base_digest,
        propose=propose,
        evaluate=evaluate,
        budget_left=budget_left,
    )
    best = result["best"] or {
        "delta": incumbent["delta"],
        "staged": incumbent["staged"],
        "evaluation": incumbent["evaluation"],
    }
    final_eval = best.get("evaluation") or incumbent["evaluation"]
    assessment = pub.assess_candidate(
        base_digest=base_digest,
        candidate_digest=(best.get("staged") or incumbent["staged"])["candidate_digest"],
        comparison=final_eval["comparison"],
        holdout=final_eval["holdout"],
        cases=final_eval["cases"],
        scan=final_eval["scan"],
    )
    candidate = {
        "delta": best.get("delta") or incumbent["delta"],
        "staged": best.get("staged") or incumbent["staged"],
        "assessment": assessment,
    }
    review = _review(
        policy=policy,
        ledger=ledger,
        deps=deps,
        job_id=job_id,
        day=day,
        lesson=prelude["lesson"],
        delta=candidate["delta"],
        assessment=assessment,
        index=counter["index"],
    )
    counter["index"] += 1
    store.save_proposal(
        job_id=job_id,
        scope_id=scope_mod.scope_id(scope),
        base_digest=base_digest,
        candidate_digest=candidate["staged"]["candidate_digest"],
        decision=assessment["decision"],
        payload={
            "delta": sd.summarize(candidate["delta"]),
            "assessment": assessment,
            "review": review,
            "search": {
                "stopped_reason": result["stopped_reason"],
                "evaluations": result["evaluations"],
                "front": [
                    {
                        "delta_id": row["delta_id"],
                        "candidate_passes": row["candidate_passes"],
                        "critical_regressions": row["critical_regressions"],
                        "changed_characters": row["changed_characters"],
                    }
                    for row in result["front"]
                ],
            },
        },
    )
    if review["verdict"] == "changes_required":
        published = {
            "published": False,
            "decision": "human_approval_required",
            "reasons": ["reviewer_changes_required"],
            "findings": review["findings"],
        }
    else:
        published = _publish_if_allowed(
            deps=deps, scope=scope, job_id=job_id, base_digest=base_digest, candidate=candidate
        )
    _finish_job(ledger, job_id, "published" if published.get("published") else assessment["decision"])
    return {
        "status": "heavy_search_complete",
        "mode": "heavy",
        "job_id": job_id,
        "lesson": prelude["lesson"],
        "lesson_record": prelude["lesson_record"],
        "search": {
            "stopped_reason": result["stopped_reason"],
            "evaluations": result["evaluations"],
            "improved": result["improved"],
            "front_size": len(result["front"]),
        },
        "delta": sd.summarize(candidate["delta"]),
        "candidate_digest": candidate["staged"]["candidate_digest"],
        "assessment": assessment,
        "review": review,
        "publish": published,
        "instruction_changed": bool(published.get("published")),
    }


PIPELINES: dict[str, Callable[..., dict[str, Any]]] = {
    "lite": run_lite,
    "lite+standard": run_standard,
    "standard": run_standard,
    "lite+standard+heavy": run_heavy,
    "standard+heavy": run_heavy,
    "lite+heavy": run_heavy,
    "heavy": run_heavy,
}


def run_learning_cycle(
    *,
    policy: dict[str, Any],
    ledger,
    store,
    scope: dict[str, Any],
    evidence: dict[str, Any],
    job_id: str,
    when,
    deps: dict[str, Any],
    dry_run: bool = False,
    search_manifest: dict[str, Any] | None = None,
    deadline: float | None = None,
    dedupe: bool = False,
) -> dict[str, Any]:
    """Dispatch one learning cycle for the profile's current mode combination."""
    pipeline = policy["pipeline"]
    base = {
        "pipeline": pipeline,
        "job_id": job_id,
        "scope": scope_mod.summarize(scope),
        "dry_run": dry_run,
    }
    if pipeline == "disabled":
        return {**base, "status": "learning_disabled", "model_calls": 0}

    routes = lp.require_routes(policy, pipeline, dry_run=dry_run)
    if not routes["ok"]:
        return {**base, "status": routes["state"], "missing": routes["missing"], "model_calls": 0}

    if not isinstance(deps, dict) or not callable(deps.get("reflector")):
        return {**base, "status": "backend_unavailable", "reason": "reflector", "model_calls": 0}
    if "standard" in pipeline or "heavy" in pipeline:
        for name in ("curator", "evaluator", "staging", "package_root"):
            if name not in deps:
                return {**base, "status": "backend_unavailable", "reason": name, "model_calls": 0}
    if "heavy" in pipeline and not callable(deps.get("reviewer")):
        return {**base, "status": "backend_unavailable", "reason": "reviewer", "model_calls": 0}

    package_root = Path(deps["package_root"]).resolve() if deps.get("package_root") else None
    base_digest = sd.package_digest(package_root) if package_root is not None else "0" * 64
    day = lp.day_key(policy, when)
    evidence_digest = scope_mod.context_digest(evidence)
    digest = manifest_digest(
        pipeline=pipeline,
        scope_id=scope_mod.scope_id(scope),
        base_digest=base_digest,
        evidence_digest=evidence_digest,
        policy=policy,
    )
    ceiling_key = lp.job_ceiling_key(pipeline)
    ceiling = 1 if pipeline == "lite" else int(policy["budget"][ceiling_key])
    if dry_run:
        return {
            **base,
            "status": "dry_run",
            "manifest_digest": digest,
            "ceiling": ceiling,
            "day": day,
            "model_calls": 0,
        }

    if dedupe and pipeline == "lite":
        existing = store.find_by_evidence(
            scope_id=scope_mod.scope_id(scope), evidence_digest=evidence_digest
        )
        if existing is not None:
            return {
                **base,
                "status": "lesson_duplicate",
                "lesson_record": existing,
                "reason": "identical evidence already produced a lesson in this scope",
                "model_calls": 0,
            }

    job_deadline = (
        deadline
        if deadline is not None
        else time.monotonic() + float(policy["budget"]["default_job_deadline_seconds"])
    )
    deps = {**deps, "deadline": job_deadline, "evidence_digest": evidence_digest}

    admission = ledger.reserve_job(
        job_id,
        manifest_digest=digest,
        scope_id=scope_mod.scope_id(scope),
        mode=pipeline,
        ceiling=ceiling,
        day=day,
        daily_cap=int(policy["budget"]["daily_model_calls"]),
    )
    if not admission.get("admitted"):
        return {**base, "status": "budget_denied", "reason": admission.get("reason"), "detail": admission, "model_calls": 0}
    if admission.get("idempotent"):
        job = ledger.job(job_id) or {}
        return {
            **base,
            "status": "job_already_recorded",
            "outcome": job.get("outcome"),
            "reason": "the same job_id and manifest were already accepted; no new model call was made",
            "model_calls": len(ledger.job_invocations(job_id)),
        }

    runner = PIPELINES.get(pipeline)
    if runner is None:  # pragma: no cover - pipeline names come from the policy
        _finish_job(ledger, job_id, "unsupported_pipeline")
        return {**base, "status": "unsupported_pipeline", "model_calls": 0}

    common = {
        "policy": policy,
        "ledger": ledger,
        "store": store,
        "deps": deps,
        "scope": scope,
        "evidence": evidence,
        "job_id": job_id,
        "day": day,
    }
    try:
        if pipeline == "lite":
            result = runner(**common)
        elif pipeline in ("lite+standard", "standard"):
            result = runner(**common, base_digest=base_digest)
        else:
            result = runner(**common, base_digest=base_digest, search_manifest=search_manifest)
    except BudgetExhausted as exc:
        _finish_job(ledger, job_id, f"budget_denied:{exc.verdict.get('reason')}")
        return {**base, "status": "budget_denied", "reason": exc.verdict.get("reason"), "detail": exc.verdict}
    except DeadlineExceeded as exc:
        _finish_job(ledger, job_id, "deadline_exceeded")
        return {
            **base,
            "status": "deadline_exceeded",
            "reason": str(exc),
            "model_calls": len(ledger.job_invocations(job_id)),
            "instruction_changed": False,
        }
    except (LearningError, sd.DeltaError, pub.PublishError, scope_mod.ScopeError) as exc:
        _finish_job(ledger, job_id, "stopped_invalid_stage")
        return {**base, "status": "stopped_invalid_stage", "reason": str(exc), "instruction_changed": False}
    except Exception as exc:  # pragma: no cover - defensive: report, never publish
        _finish_job(ledger, job_id, "stopped_error")
        return {**base, "status": "stopped_error", "reason": f"{type(exc).__name__}: {exc}", "instruction_changed": False}

    result["model_calls"] = len(ledger.job_invocations(job_id))
    return {**base, **result}
