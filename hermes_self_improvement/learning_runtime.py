"""Runtime wiring for the Skillcraft learning lane.

This module is the only place that touches the host: it reads the profile
config, creates the profile-local ledger and lesson store, and adapts the
existing tool-free role agents into the reflector/curator/task callables the
pipelines expect.

Nothing here dispatches a model on import, and every mutating action requires an
explicit ``allow_model_calls`` acknowledgement from the caller.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import learning_budget as lb
from . import learning_evaluator as lev
from . import learning_modes as lm
from . import learning_policy as lp
from . import learning_store as ls
from . import skill_delta as sd
from . import skill_exposure as se
from . import skill_scope as ss

ACTIONS = ("status", "check", "lesson", "learn", "report", "exposure")
MODEL_ACTIONS = ("lesson", "learn")
_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


class RuntimeErrorLearning(PermissionError):
    """Raised when a caller asks for model work without acknowledging the spend."""


def state_dir(home: Any = None) -> Path:
    base = lp._home(home) / "self-improvement" / "learning"
    base.mkdir(parents=True, exist_ok=True)
    return base


def policy_for(home: Any = None, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return lp.load_policy(home, overrides=overrides)


def _ledger(home: Any = None) -> lb.Ledger:
    return lb.Ledger(state_dir(home) / "budget.sqlite3")


def _store(home: Any = None) -> ls.LessonStore:
    return ls.LessonStore(state_dir(home) / "lessons.sqlite3")


def _exposure(home: Any = None) -> se.ExposureStore:
    return se.ExposureStore(state_dir(home) / "exposure.sqlite3")


def _role_runner(
    role: str, home: Any = None, route: dict[str, str] | None = None
) -> Callable[[str], tuple[str, dict[str, Any]]]:
    """Adapt one tool-free role agent into a text callable.

    ``route`` is applied to the role config so the dispatched provider/model/effort
    is the same one the ledger recorded; a policy route is never decorative.
    """
    from .config import load_config
    from .constrained_agent import run_tool_free_role_agent

    config = load_config()
    if route:
        model_cfg = dict(config.get("model") or {})
        role_cfg = dict(model_cfg.get(role) or {})
        role_cfg["provider"] = route["provider"]
        role_cfg["model"] = route["model"]
        extra = dict(role_cfg.get("extra_body") or {})
        extra["reasoning"] = {"effort": route["reasoning"]}
        role_cfg["extra_body"] = extra
        model_cfg[role] = role_cfg
        config = {**config, "model": model_cfg}

    def run(prompt: str) -> tuple[str, dict[str, Any]]:
        result = run_tool_free_role_agent(
            role=role,
            user_message=prompt,
            system_message="Answer with the requested JSON object only. No prose, no tools.",
            config=config,
            max_iterations=1,
        )
        text = result.get("final_response") or result.get("content") or ""
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        return text, usage

    return run


def _model_callable(role: str) -> Callable[[dict[str, Any], dict[str, str]], dict[str, Any]]:
    def call(payload: dict[str, Any], route: dict[str, str]) -> dict[str, Any]:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise lm.LearningError("model payload has no prompt")
        text, usage = _role_runner(role, route=route)(prompt)
        return {"text": text, "usage": usage, "route": route}

    return call


def _frontmatter(text: str) -> dict[str, Any]:
    match = _FRONTMATTER.match(text or "")
    if not match:
        return {}
    try:
        import yaml
    except Exception:  # pragma: no cover - the Hermes runtime ships PyYAML
        return {}
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def skill_structure(package_root: Any) -> dict[str, Any]:
    root = Path(package_root).expanduser().resolve()
    skill_file = root / "SKILL.md"
    report: dict[str, Any] = {
        "package_root": str(root),
        "exists": skill_file.is_file(),
        "name": None,
        "description": None,
        "version": None,
        "support_links": [],
        "missing_links": [],
        "digest": None,
    }
    if not skill_file.is_file():
        return report
    meta = _frontmatter(skill_file.read_text(encoding="utf-8"))
    report["name"] = meta.get("name")
    report["description"] = meta.get("description")
    report["version"] = meta.get("version")
    links = sorted(set(re.findall(r"\]\((references/[^)\s]+)\)", skill_file.read_text(encoding="utf-8"))))
    report["support_links"] = links
    report["missing_links"] = [link for link in links if not (root / link).is_file()]
    report["digest"] = sd.package_digest(root)
    return report


def status(home: Any = None) -> dict[str, Any]:
    policy = policy_for(home)
    day = lp.day_key(policy, datetime.now(timezone.utc))
    ledger = _ledger(home)
    budget = ledger.status(
        day,
        daily_cap=int(policy["budget"]["daily_model_calls"]),
        lite_daily_cap=int(policy["budget"]["lite_daily_model_calls"]),
    )
    store = _store(home)
    return {
        "ok": True,
        "action": "status",
        "policy": lp.summarize(policy),
        "routes": lp.require_routes(policy, policy["pipeline"]),
        "budget": budget,
        "lessons": len(store.recent_lessons(limit=200)),
        "recent_lessons": [
            {"job_id": row["job_id"], "status": row["status"], "observation": row["lesson"].get("observation")}
            for row in store.recent_lessons(limit=5)
        ],
        "model_calls": 0,
    }


def check(home: Any = None, *, package_root: Any = None) -> dict[str, Any]:
    policy = policy_for(home)
    result: dict[str, Any] = {
        "ok": True,
        "action": "check",
        "policy": lp.summarize(policy),
        "routes": lp.require_routes(policy, policy["pipeline"]),
        "model_calls": 0,
        "scope": "static checks only; no model call, no package write",
    }
    if package_root is not None:
        structure = skill_structure(package_root)
        result["skill"] = structure
        result["ok"] = bool(structure["exists"] and not structure["missing_links"])
    return result


def _require_model_authorization(args: dict[str, Any]) -> None:
    if args.get("allow_model_calls") is not True:
        raise RuntimeErrorLearning(
            "Paid/model inference needs explicit allow_model_calls=true"
        )


def _scope_for(home: Any, args: dict[str, Any]) -> dict[str, Any]:
    cwd = args.get("cwd")
    if cwd is None:
        cwd = lp._home(home)
    return ss.resolve_scope(cwd, home)


def _evidence(args: dict[str, Any]) -> dict[str, Any]:
    evidence = args.get("evidence")
    if evidence is None:
        return {}
    if isinstance(evidence, str):
        path = Path(evidence).expanduser()
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            try:
                loaded = json.loads(text)
                return loaded if isinstance(loaded, dict) else {"text": text}
            except json.JSONDecodeError:
                return {"text": text[:20_000]}
        return {"text": evidence[:20_000]}
    if isinstance(evidence, dict):
        return evidence
    raise lm.LearningError("evidence must be an object or a path to JSON/text")


def _build_deps(home: Any, *, package_root: Path, cases_path: Any) -> dict[str, Any]:
    deps: dict[str, Any] = {
        "reflector": _model_callable("planner"),
        "curator": _model_callable("editor"),
        "reviewer": _model_callable("planner"),
        "package_root": package_root,
        "home": lp._home(home),
    }

    def staging(base: Path) -> Path:
        import shutil
        import tempfile

        target = Path(tempfile.mkdtemp(prefix="skillcraft-staging-", dir=str(state_dir(home))))
        destination = target / "package"
        shutil.copytree(base, destination)
        return destination

    deps["staging"] = staging

    if cases_path is not None:
        cases = lev.load_cases(Path(cases_path).expanduser().read_text(encoding="utf-8"))
        candidate_holder: dict[str, str] = {}

        def evaluator(payload: dict[str, Any], route: dict[str, str]) -> dict[str, Any]:
            candidate_root = Path(payload["candidate_root"])
            base_root = Path(payload["base_root"])
            baseline_text = (base_root / "SKILL.md").read_text(encoding="utf-8")
            candidate_text = (candidate_root / "SKILL.md").read_text(encoding="utf-8")
            candidate_holder["text"] = candidate_text
            task_runner = _role_runner("evaluator", route=route)

            def run(prompt: str) -> str:
                text, _usage = task_runner(prompt)
                return text

            scan_findings = sd.secret_hits(candidate_text)
            return lev.evaluate(
                cases=cases,
                baseline_text=baseline_text,
                candidate_text=candidate_text,
                run=run,
                scan={"status": "fail" if scan_findings else "pass", "findings": scan_findings},
            )

        deps["evaluator"] = evaluator
    return deps


def _native_writer(home: Any) -> Callable[..., dict[str, Any]] | None:
    """Automatic writes stay unavailable unless the host exposes a writer.

    An explicit ``_learning_native_writer`` callable is an operator decision and
    wins. Otherwise the host's own skill tool is used, but only when the profile
    policy enables ``auto_publish_enabled``: a writer that exists is not yet
    permission to write.
    """
    from . import learning_writer as lwd
    from .config import load_config

    config = load_config() or {}
    writer = config.get("_learning_native_writer")
    if callable(writer):
        return writer
    if not policy_for(home).get("budget", {}).get("auto_publish_enabled"):
        return None
    writer = lwd.build_writer()
    return writer if callable(writer) else None


def _native_writer_reason(home: Any) -> str | None:
    """``None`` when a writer is available, else the reason it is not."""
    from . import learning_writer as lwd
    from .config import load_config

    config = load_config() or {}
    if callable(config.get("_learning_native_writer")):
        return None
    if not policy_for(home).get("budget", {}).get("auto_publish_enabled"):
        return "auto_publish_disabled: set learning_budget.auto_publish_enabled to true"
    if callable(lwd.build_writer()):
        return None
    return lwd.diagnose()


def _cycle_id(args: dict[str, Any], action: str) -> str:
    raw = args.get("cycle_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{action}-{stamp}"


def run_action(args: dict[str, Any], *, home: Any = None) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {"ok": False, "error": "arguments must be an object"}
    action = args.get("action")
    if action not in ACTIONS:
        return {"ok": False, "error": f"action must be one of {', '.join(ACTIONS)}"}
    try:
        if action == "status":
            return status(home)
        if action == "check":
            return check(home, package_root=args.get("package"))
        if action == "report":
            store = _store(home)
            return {
                "ok": True,
                "action": "report",
                "lessons": store.recent_lessons(limit=20),
                "proposals": store.proposals(limit=20),
                "recurring": store.recurring(min_occurrences=2),
                "model_calls": 0,
            }
        if action == "exposure":
            return _exposure_action(args, home)
        _require_model_authorization(args)
        if not isinstance(args.get("package"), str):
            return {"ok": False, "error": "package (skill directory) is required"}
        package_root = Path(args["package"]).expanduser().resolve()
        structure = skill_structure(package_root)
        if not structure["exists"] or structure["missing_links"]:
            return {"ok": False, "error": "skill package is not loadable", "skill": structure}
        scope = _scope_for(home, args)
        policy = policy_for(home)
        if action == "lesson" and "standard" in policy["pipeline"] and policy["pipeline"] != "lite":
            policy = lp.clone(policy)
            policy["modes"] = {"lite_enabled": True, "standard_enabled": False, "heavy_enabled": False}
            policy["pipeline"] = "lite"
            policy["enabled_modes"] = ("lite",)
        deps = _build_deps(home, package_root=package_root, cases_path=args.get("cases"))
        writer = _native_writer(home)
        if writer is not None:
            deps["native_writer"] = writer
        else:
            deps["native_writer_error"] = _native_writer_reason(home)
        result = lm.run_learning_cycle(
            policy=policy,
            ledger=_ledger(home),
            store=_store(home),
            scope=scope,
            evidence=_evidence(args),
            job_id=_cycle_id(args, action),
            when=datetime.now(timezone.utc),
            deps=deps,
            dry_run=bool(args.get("dry_run")),
            search_manifest=args.get("search") if isinstance(args.get("search"), dict) else None,
            dedupe=bool(args.get("dedupe")),
        )
        result["ok"] = result.get("status") not in (
            "stopped_error",
            "stopped_invalid_stage",
            "backend_unavailable",
            "budget_denied",
            "route_unconfigured",
            "deadline_exceeded",
        )
        result["auto_write_available"] = writer is not None
        return result
    except RuntimeErrorLearning as exc:
        return {"ok": False, "error": str(exc), "action": action}
    except (ss.ScopeError, lb.BudgetError, ls.StoreError, se.ExposureError, lev.EvaluatorError, lm.LearningError) as exc:
        return {"ok": False, "error": str(exc), "action": action}
    except Exception as exc:  # pragma: no cover - never leak a traceback into a tool result
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "action": action}


def _exposure_action(args: dict[str, Any], home: Any) -> dict[str, Any]:
    store = _exposure(home)
    if args.get("record") is not None:
        record = args["record"]
        if not isinstance(record, dict):
            return {"ok": False, "error": "record must be an object"}
        return {
            "ok": True,
            "action": "exposure",
            **store.record_exposure(
                event_id=str(record.get("event_id")),
                skill_name=str(record.get("skill_name")),
                skill_digest=str(record.get("skill_digest")),
                scope_id=str(record.get("scope_id") or "unknown"),
                session_id=str(record.get("session_id") or "unknown"),
            ),
        }
    if args.get("bind") is not None:
        bind = args["bind"]
        if not isinstance(bind, dict):
            return {"ok": False, "error": "bind must be an object"}
        return {
            "ok": True,
            "action": "exposure",
            **store.bind_confirmed_outcome(
                str(bind.get("exposure_id")),
                outcome_id=str(bind.get("outcome_id")),
                kind=str(bind.get("kind")),
                passed=bind.get("passed"),
                evidence_ids=bind.get("evidence_ids") or [],
                expected_digest=str(bind.get("expected_digest")),
            ),
        }
    skill_name = args.get("skill_name")
    if not isinstance(skill_name, str) or not skill_name.strip():
        return {"ok": False, "error": "skill_name is required to summarize exposures"}
    return {"ok": True, "action": "exposure", **store.summarize(skill_name=skill_name)}


def format_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)


def learning_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Tool entry point; returns a dict the host serialises."""
    return run_action(args or {})
