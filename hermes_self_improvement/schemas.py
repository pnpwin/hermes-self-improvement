from __future__ import annotations

STRING = {"type": "string"}
CONFIG_PATH_PROPERTY = {"type": "string", "description": "Explicit config YAML path; same precedence as CLI --config."}

SELF_IMPROVEMENT_STATUS_SCHEMA = {
    "name": "self_improvement_status",
    "description": "Show hermes-self-improvement plugin status. Read-only.",
    "parameters": {
        "type": "object",
        "properties": {"config_path": CONFIG_PATH_PROPERTY},
    },
}

SELF_IMPROVEMENT_REPORT_SCHEMA = {
    "name": "self_improvement_report",
    "description": "Generate a self-improvement analysis report. Read-only; does not mutate targets.",
    "parameters": {
        "type": "object",
        "properties": {
            "config_path": CONFIG_PATH_PROPERTY,
            "since_hours": {"type": "integer", "default": 24},
        },
    },
}

SELF_IMPROVEMENT_IMPROVE_SCHEMA = {
    "name": "self_improvement_improve",
    "description": "Run self-improvement. Mutates by default; dry_run=true previews without mutation.",
    "parameters": {
        "type": "object",
        "properties": {
            "config_path": CONFIG_PATH_PROPERTY,
            "since_hours": {"type": "integer", "default": 24},
            "dry_run": {"type": "boolean", "default": False},
        },
    },
}

SELF_IMPROVEMENT_CALIBRATE_SCHEMA = {
    "name": "self_improvement_calibrate",
    "description": "Calibrate evaluator prompts/rubrics. Mutates active evaluator state by default when gates pass; dry_run=true previews.",
    "parameters": {
        "type": "object",
        "properties": {
            "config_path": CONFIG_PATH_PROPERTY,
            "dry_run": {"type": "boolean", "default": False},
            "candidate_set_artifact_path": {"type": "string", "description": "Optional explicit dry-run overlay candidate-set artifact path to reuse during execute mode."},
        },
    },
}

SELF_IMPROVEMENT_LEARNING_SCHEMA = {
    "name": "self_improvement_learning",
    "description": (
        "Skillcraft learning lane: status/check/report are read-only; lesson/learn dispatch "
        "model calls only with allow_model_calls=true and write a skill package only through "
        "the host's native writer. Lite saves a lesson, Standard proposes a bounded delta, "
        "Heavy runs a capped search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "check", "lesson", "learn", "report", "exposure"],
            },
            "config_path": CONFIG_PATH_PROPERTY,
            "package": {"type": "string", "description": "Absolute path to the skill directory under test."},
            "cases": {"type": "string", "description": "Path to a frozen JSON case file for Standard/Heavy."},
            "evidence": {"type": "string", "description": "Trial evidence: an object, JSON text or a path."},
            "cycle_id": {"type": "string", "description": "Idempotency key; the same id never spends twice."},
            "allow_model_calls": {"type": "boolean", "default": False},
            "dry_run": {"type": "boolean", "default": False},
            "cwd": {"type": "string", "description": "Project directory; decides project vs home scope."},
            "skill_name": {"type": "string", "description": "Skill name for an exposure summary."},
            "record": {"type": "object", "description": "Record one exposure event."},
            "bind": {"type": "object", "description": "Bind a confirmed outcome to an exposure."},
            "search": {"type": "object", "description": "Bounded Heavy search manifest."},
        },
        "required": ["action"],
    },
}

SELF_IMPROVEMENT_TOOL_SPECS = (
    ("self_improvement_status", SELF_IMPROVEMENT_STATUS_SCHEMA),
    ("self_improvement_report", SELF_IMPROVEMENT_REPORT_SCHEMA),
    ("self_improvement_improve", SELF_IMPROVEMENT_IMPROVE_SCHEMA),
    ("self_improvement_calibrate", SELF_IMPROVEMENT_CALIBRATE_SCHEMA),
    ("self_improvement_learning", SELF_IMPROVEMENT_LEARNING_SCHEMA),
)
