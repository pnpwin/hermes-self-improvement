from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hermes_self_improvement import learning_policy as lp


ALL_EIGHT = [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]


def _flags(lite: bool, standard: bool, heavy: bool) -> dict:
    return {
        "lite_enabled": lite,
        "standard_enabled": standard,
        "heavy_enabled": heavy,
    }


@pytest.mark.parametrize("lite,standard,heavy", ALL_EIGHT)
def test_every_combination_has_one_pipeline(lite, standard, heavy):
    policy = lp.build_policy(_flags(lite, standard, heavy))
    pipeline = policy["pipeline"]
    expected_parts = [
        name
        for name, on in (("lite", lite), ("standard", standard), ("heavy", heavy))
        if on
    ]
    assert pipeline == ("+".join(expected_parts) or "disabled")
    assert policy["enabled_modes"] == tuple(expected_parts)


def test_defaults_are_lite_and_standard_only():
    policy = lp.build_policy({})
    assert policy["modes"] == {
        "lite_enabled": True,
        "standard_enabled": True,
        "heavy_enabled": False,
    }
    assert policy["pipeline"] == "lite+standard"


def test_string_false_is_rejected():
    with pytest.raises(lp.LearningPolicyError):
        lp.normalize_mode_flags({"heavy_enabled": "false"})


def test_integer_and_null_are_rejected():
    for bad in (1, 0, None, "true", [], {}):
        with pytest.raises(lp.LearningPolicyError):
            lp.normalize_mode_flags({"lite_enabled": bad})


def test_unknown_mode_flag_is_rejected():
    with pytest.raises(lp.LearningPolicyError):
        lp.build_policy({"mode": "heavy"})


def test_budget_negative_and_float_are_rejected():
    for bad in (-1, 0, 2.5, "8", True):
        with pytest.raises(lp.LearningPolicyError):
            lp.normalize_budget({"daily_model_calls": bad})


def test_lite_cap_cannot_exceed_daily_cap():
    with pytest.raises(lp.LearningPolicyError):
        lp.normalize_budget({"daily_model_calls": 4, "lite_daily_model_calls": 8})


def test_nan_and_infinity_are_rejected():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(lp.LearningPolicyError):
            lp.normalize_budget({"standard_job_model_calls": bad})


def test_timezone_must_exist():
    with pytest.raises(lp.LearningPolicyError):
        lp.normalize_budget({"timezone": "Mars/Olympus"})


def test_route_requires_exactly_three_fields():
    good = {"provider": "p", "model": "m", "reasoning": "low"}
    assert lp.route_spec(good) == good
    for bad in (
        {"provider": "p", "model": "m"},
        {**good, "api_key": "secret"},
        {**good, "reasoning": "turbo"},
        {"provider": "", "model": "m", "reasoning": "low"},
        {"provider": " p", "model": "m", "reasoning": "low"},
    ):
        with pytest.raises(lp.LearningPolicyError):
            lp.route_spec(bad)


def test_routes_are_null_until_configured():
    policy = lp.build_policy({})
    assert set(policy["routes"]) == set(lp.ROUTE_ROLES)
    assert all(value is None for value in policy["routes"].values())


def test_enabled_stage_without_route_fails_closed():
    policy = lp.build_policy({})
    verdict = lp.require_routes(policy, policy["pipeline"])
    assert verdict["ok"] is False
    assert verdict["state"] == "route_unconfigured"
    assert verdict["missing"] == ["reflector", "curator", "task"]


def test_disabled_pipeline_is_not_a_ready_state():
    policy = lp.build_policy(_flags(False, False, False))
    assert policy["pipeline"] == "disabled"
    assert lp.require_routes(policy, policy["pipeline"])["state"] == "learning_disabled"


def test_all_routes_configured_is_ready_and_heavy_adds_reviewer():
    route = {"provider": "p", "model": "m", "reasoning": "medium"}
    policy = lp.build_policy(
        {**_flags(False, False, True), "learning_routes": {role: route for role in lp.ROUTE_ROLES}}
    )
    assert lp.require_routes(policy, policy["pipeline"])["ok"] is True
    assert "reviewer" in lp.job_role_sequence(policy["pipeline"])


def test_heavy_ceiling_is_used_when_heavy_is_enabled():
    assert lp.job_ceiling_key("lite") == "standard_job_model_calls"
    assert lp.job_ceiling_key("standard+heavy") == "heavy_job_model_calls"


def test_day_key_uses_configured_timezone_not_process_locale():
    policy = lp.build_policy({"learning_budget": {"timezone": "Europe/Moscow"}})
    # 22:30 UTC on 2026-11-08 is already 01:30 MSK on 2026-11-09.
    when = datetime(2026, 11, 8, 22, 30, tzinfo=timezone.utc)
    assert lp.day_key(policy, when) == "2026-11-09"
    utc_policy = lp.build_policy({"learning_budget": {"timezone": "UTC"}})
    assert lp.day_key(utc_policy, when) == "2026-11-08"


def test_policy_reads_settings_leaves_and_never_payload_routes(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "plugins:\n"
        "  entries:\n"
        "    hermes-skillcraft:\n"
        "      settings:\n"
        "        lite_enabled: true\n"
        "        standard_enabled: false\n"
        "        heavy_enabled: false\n",
        encoding="utf-8",
    )
    policy = lp.load_policy(tmp_path)
    assert policy["pipeline"] == "lite"
    assert policy["budget"]["daily_model_calls"] == 24


def test_policy_ignores_settings_owned_by_the_trial_runner(tmp_path):
    """``settings`` is shared: ``primary``/``fallback_chain`` belong to the trial runner.

    Rejecting them made every documented configuration break the learning lane.
    A typo inside the policy namespace must still be refused.
    """
    route = {"provider": "eliza", "model": "~deepseek/deepseek-flash-latest", "reasoning": "low"}
    policy = lp.build_policy(
        {
            "primary": route,
            "fallback_chain": [route],
            "lite_enabled": True,
            "standard_enabled": False,
            "heavy_enabled": False,
            "learning_routes": {"reflector": route},
        }
    )
    assert policy["pipeline"] == "lite"
    assert policy["routes"]["reflector"] == route
    with pytest.raises(lp.LearningPolicyError, match="learning_bugdet"):
        lp.build_policy({"primary": route, "learning_bugdet": {"daily_model_calls": 4}})


def test_summary_is_compact_and_secret_free():
    route = {"provider": "p", "model": "m", "reasoning": "low"}
    policy = lp.build_policy({"learning_routes": {"reflector": route}})
    summary = lp.summarize(policy)
    assert summary["routes_configured"] == ["reflector"]
    assert "api_key" not in str(summary)
    assert "m" not in summary["routes_missing"]


def test_auto_publish_is_off_by_default_and_must_be_a_real_boolean():
    assert lp.DEFAULT_BUDGET["auto_publish_enabled"] is False
    assert lp.normalize_budget({"auto_publish_enabled": True})["auto_publish_enabled"] is True
    with pytest.raises(lp.LearningPolicyError, match="auto_publish_enabled"):
        lp.normalize_budget({"auto_publish_enabled": "true"})
