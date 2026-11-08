from __future__ import annotations

import threading

import pytest

from hermes_self_improvement.learning_budget import BudgetError, Ledger


ROUTE = {"provider": "p", "model": "m", "reasoning": "low"}
DAY = "2026-11-09"


def _ledger(tmp_path, clock=None):
    return Ledger(tmp_path / "state.sqlite3", **({"clock": clock} if clock else {}))


def _open_job(ledger, job_id="job-1", *, ceiling=14, daily=24, digest="d1"):
    return ledger.reserve_job(
        job_id,
        manifest_digest=digest,
        scope_id="scope-1",
        mode="standard",
        ceiling=ceiling,
        day=DAY,
        daily_cap=daily,
    )


def test_lite_daily_cap_denies_ninth_call_before_dispatch(tmp_path):
    ledger = _ledger(tmp_path)
    _open_job(ledger, ceiling=20)
    admitted = 0
    for index in range(9):
        job_id = f"lite-{index}"
        assert ledger.reserve_job(
            job_id,
            manifest_digest=job_id,
            scope_id="scope-1",
            mode="lite",
            ceiling=1,
            day=DAY,
            daily_cap=24,
        )["admitted"] is True
        verdict = ledger.reserve_call(
            job_id,
            f"{job_id}-inv",
            role="reflector",
            route=ROUTE,
            day=DAY,
            daily_cap=24,
            lite_daily_cap=8,
            lite=True,
        )
        if verdict["admitted"]:
            admitted += 1
    assert admitted == 8
    assert verdict == {
        "admitted": False,
        "reason": "lite_daily_cap",
        "used_lite_today": 8,
        "lite_daily_cap": 8,
    }


def test_concurrent_reservations_never_exceed_cap(tmp_path):
    ledger = _ledger(tmp_path)
    opened = _open_job(ledger, ceiling=5, daily=5)
    assert opened["admitted"] is True, opened
    assert ledger.job("job-1") is not None
    admitted: list[str] = []
    verdicts: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            local = Ledger(tmp_path / "state.sqlite3")
            verdict = local.reserve_call(
                "job-1",
                f"inv-{index}",
                role="task",
                route=ROUTE,
                day=DAY,
                daily_cap=5,
                lite_daily_cap=5,
            )
            with lock:
                verdicts.append(verdict)
            if verdict["admitted"]:
                with lock:
                    admitted.append(verdict["invocation_id"])
        except Exception as exc:  # pragma: no cover - surfaced by the assertion below
            with lock:
                errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(admitted) == 5, verdicts
    assert len(set(admitted)) == 5
    assert ledger.status(DAY, daily_cap=5, lite_daily_cap=5)["used_model_calls"] == 5


def test_crash_after_dispatch_keeps_the_slot_and_is_visible(tmp_path):
    ledger = _ledger(tmp_path)
    _open_job(ledger, ceiling=3)
    first = ledger.reserve_call(
        "job-1", "inv-1", role="task", route=ROUTE, day=DAY, daily_cap=24, lite_daily_cap=8
    )
    assert first["admitted"] is True
    # The client dies here: no finish_call. Re-reserving the same attempt is idempotent.
    retry = ledger.reserve_call(
        "job-1", "inv-1", role="task", route=ROUTE, day=DAY, daily_cap=24, lite_daily_cap=8
    )
    assert retry == {
        "admitted": True,
        "idempotent": True,
        "invocation_id": "inv-1",
        "status": "reserved",
    }
    assert [row["invocation_id"] for row in ledger.pending_invocations("job-1")] == ["inv-1"]
    assert ledger.status(DAY, daily_cap=24, lite_daily_cap=8)["used_model_calls"] == 1


def test_job_ceiling_is_enforced_independently_of_daily_cap(tmp_path):
    ledger = _ledger(tmp_path)
    _open_job(ledger, ceiling=2, daily=100)
    for index in range(3):
        verdict = ledger.reserve_call(
            "job-1", f"inv-{index}", role="task", route=ROUTE, day=DAY, daily_cap=100, lite_daily_cap=8
        )
    assert verdict["admitted"] is False
    assert verdict["reason"] == "job_ceiling"


def test_timeout_after_dispatch_does_not_return_the_slot(tmp_path):
    ledger = _ledger(tmp_path)
    _open_job(ledger, ceiling=2)
    ledger.reserve_call("job-1", "inv-1", role="task", route=ROUTE, day=DAY, daily_cap=24, lite_daily_cap=8)
    ledger.finish_call("inv-1", status="timeout", usage={"total_tokens": None})
    ledger.reserve_call("job-1", "inv-2", role="task", route=ROUTE, day=DAY, daily_cap=24, lite_daily_cap=8)
    blocked = ledger.reserve_call(
        "job-1", "inv-3", role="task", route=ROUTE, day=DAY, daily_cap=24, lite_daily_cap=8
    )
    assert blocked["reason"] == "job_ceiling"
    assert ledger.status(DAY, daily_cap=24, lite_daily_cap=8)["used_model_calls"] == 2


def test_job_admission_requires_daily_remainder_for_the_whole_reservation(tmp_path):
    ledger = _ledger(tmp_path)
    for index in range(16):
        ledger.reserve_job(
            f"filler-{index}",
            manifest_digest=f"filler-{index}",
            scope_id="scope-1",
            mode="lite",
            ceiling=1,
            day=DAY,
            daily_cap=24,
        )
        ledger.reserve_call(
            f"filler-{index}",
            f"filler-inv-{index}",
            role="reflector",
            route=ROUTE,
            day=DAY,
            daily_cap=24,
            lite_daily_cap=24,
            lite=True,
        )
    verdict = ledger.reserve_job(
        "heavy-1",
        manifest_digest="heavy",
        scope_id="scope-1",
        mode="heavy",
        ceiling=14,
        day=DAY,
        daily_cap=24,
    )
    assert verdict["admitted"] is False
    assert verdict["reason"] == "daily_cap"
    assert verdict["used_today"] == 16


def test_same_job_id_with_another_manifest_is_a_conflict(tmp_path):
    ledger = _ledger(tmp_path)
    first = _open_job(ledger, digest="digest-a")
    assert first["admitted"] is True
    second = _open_job(ledger, digest="digest-b")
    assert second == {"admitted": False, "reason": "job_id_conflict", "job_id": "job-1"}
    repeat = _open_job(ledger, digest="digest-a")
    assert repeat["admitted"] is True and repeat["idempotent"] is True


def test_closed_job_rejects_new_calls_and_close_is_idempotent(tmp_path):
    ledger = _ledger(tmp_path)
    _open_job(ledger)
    ledger.close_job("job-1", outcome="lesson_saved")
    verdict = ledger.reserve_call(
        "job-1", "inv-1", role="reflector", route=ROUTE, day=DAY, daily_cap=24, lite_daily_cap=8
    )
    assert verdict["reason"] == "job_closed"
    again = ledger.close_job("job-1", outcome="keep_incumbent")
    assert again == {"closed": True, "idempotent": True, "outcome": "lesson_saved"}


def test_usage_is_stored_verbatim_including_unknown_values(tmp_path):
    ledger = _ledger(tmp_path)
    _open_job(ledger)
    ledger.reserve_call("job-1", "inv-1", role="task", route=ROUTE, day=DAY, daily_cap=24, lite_daily_cap=8)
    usage = {"input_tokens": 10, "output_tokens": None, "cache_read_tokens": None}
    ledger.finish_call("inv-1", status="completed", usage=usage)
    rows = ledger.job_invocations("job-1")
    assert rows[0]["status"] == "completed"
    assert '"output_tokens": null' in rows[0]["usage_json"]


def test_malformed_input_is_rejected_before_any_write(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(BudgetError):
        ledger.reserve_job("", manifest_digest="d", scope_id="s", mode="lite", ceiling=1, day=DAY, daily_cap=24)
    with pytest.raises(BudgetError):
        ledger.reserve_job("job", manifest_digest="d", scope_id="s", mode="lite", ceiling=-1, day=DAY, daily_cap=24)
    with pytest.raises(BudgetError):
        ledger.finish_call("inv", status="maybe")


def test_status_reports_local_scope_not_provider_quota(tmp_path):
    ledger = _ledger(tmp_path)
    status = ledger.status(DAY, daily_cap=24, lite_daily_cap=8)
    assert status["used_model_calls"] == 0
    assert status["remaining_model_calls"] == 24
    assert "not a provider subscription quota" in status["scope"]


# -- M2: concurrency and stale reservations ------------------------------


def _reserve(ledger, invocation_id, *, job_id="job-1", max_concurrent=None, stale=1800):
    return ledger.reserve_call(
        job_id,
        invocation_id,
        role="reflector",
        route=ROUTE,
        day=DAY,
        daily_cap=24,
        lite_daily_cap=8,
        max_concurrent=max_concurrent,
        stale_after_seconds=stale,
    )


def test_concurrency_limit_refuses_a_second_simultaneous_call(tmp_path):
    ledger = _ledger(tmp_path)
    _open_job(ledger, ceiling=6)
    first = _reserve(ledger, "inv-1", max_concurrent=1)
    assert first["admitted"] is True
    second = _reserve(ledger, "inv-2", max_concurrent=1)
    assert second == {"admitted": False, "reason": "concurrency_limit", "active": 1, "max_concurrent": 1}
    ledger.finish_call("inv-1", status="completed")
    third = _reserve(ledger, "inv-2", max_concurrent=1)
    assert third["admitted"] is True


def test_stale_reservation_frees_the_slot_but_keeps_the_spend(tmp_path):
    clock = {"now": 1_000.0}
    ledger = _ledger(tmp_path, clock=lambda: clock["now"])
    _open_job(ledger, ceiling=6)
    assert _reserve(ledger, "inv-1")["admitted"] is True

    clock["now"] += 120
    assert _reserve(ledger, "inv-2", max_concurrent=1)["reason"] == "concurrency_limit"

    clock["now"] += 120  # inv-1 is now 240s old
    reaped = ledger.reap_stale(older_than_seconds=1800)
    assert reaped["reaped"] == 0  # 240s is younger than the default stale window

    clock["now"] += 1800
    assert ledger.reap_stale(older_than_seconds=1800)["reaped"] == 1
    assert ledger.status(DAY, daily_cap=24, lite_daily_cap=8)["used_model_calls"] == 1
    assert _reserve(ledger, "inv-3", max_concurrent=1)["admitted"] is True


def test_stale_window_must_be_a_positive_integer(tmp_path):
    ledger = _ledger(tmp_path)
    _open_job(ledger)
    with pytest.raises(BudgetError):
        _reserve(ledger, "inv-1", stale=0)
    with pytest.raises(BudgetError):
        _reserve(ledger, "inv-1", max_concurrent=0)
