# Skillcraft learning lane

The learning lane is the only path in this plugin that may change a skill package
automatically. It exists because Skillcraft (`hermes-skillcraft`) must not become a
second owner of automatic changes: Skillcraft selects a mode and a scope, and this
plugin executes, accounts, gates and (optionally) writes.

## Modules

| Module | Responsibility |
|---|---|
| `learning_policy.py` | canonical mode flags, budget defaults, explicit routes, day key, fail-closed route check |
| `learning_budget.py` | SQLite ledger: job admission, per-call reservation before dispatch, daily and Lite caps |
| `learning_store.py` | lessons (`observation`/`validated`) and proposals with their decision |
| `learning_modes.py` | Lite / Standard / Heavy pipelines, manifest digest, idempotency |
| `learning_evaluator.py` | bounded text-only paired evaluation with explicit checks |
| `learning_publish.py` | pure promotion gate and the native-writer hand-off |
| `learning_search.py` | capped Pareto search for Heavy |
| `skill_scope.py` | project vs home scope, trust requirement, context digest |
| `skill_delta.py` | typed bounded delta validation and staged application |
| `skill_exposure.py` | exposure vs bound outcome telemetry, per version |
| `learning_runtime.py` | the only host-touching layer: config, ledger paths, role-agent adapters, tool entry point |

## Invariants

1. One pipeline per cycle, chosen from the eight mode combinations.
2. A mode flag is a JSON boolean or the cycle stops; quoted strings are refused.
3. Every model call is reserved before dispatch. A crash or timeout keeps the
   slot, because the provider may already have billed it.
4. `max_concurrent_model_calls` is enforced by the ledger; an abandoned
   reservation is reaped to `ambiguous` after the stale window, which frees the
   concurrency slot but never refunds the daily counter.
5. A job has a wall-clock deadline from `default_job_deadline_seconds`; once it
   passes, the next stage is not dispatched and the job ends `deadline_exceeded`.
4. A delta is small and typed: at most 8 operations, 3 files, 8000 characters,
   bound to the base digest it was produced from.
5. A publish needs an `auto_publish` decision *and* a native writer. Without the
   writer the cycle stops at `human_approval_required`; nothing is written.
6. Heavy adds one bounded reviewer call. `changes_required` blocks publication
   but does not rewrite the gate decision; the finding is preserved as evidence.
7. A lesson never becomes an instruction without a Standard cycle and paired
   evidence.
8. Budgets are local to the profile and are never described as provider quota.
9. `--dedupe` skips a Lite cycle whose evidence already produced a lesson in the
   same scope; without the flag every cycle is charged, because a repeated task
   is a new observation.
10. `report` groups repeated observations (`count >= 2`) as a signal that a
    paired cycle is due; a group is never an automatic instruction change.

## Commands

```bash
hermes self-improvement learning --action status
hermes self-improvement learning --action check --package /path/to/skill
hermes self-improvement learning --action lesson --evidence evidence.json --allow-model-calls
hermes self-improvement learning --action learn --package /path/to/skill --cases cases.json \
  --cycle-id cycle-1 --allow-model-calls
hermes self-improvement learning --action report
```

The same surface is available as the `self_improvement_learning` tool, which is
what `hermes-skillcraft` calls.

## Configuration

Configured by the owning profile under
`plugins.entries.hermes-skillcraft.settings`:

* `lite_enabled`, `standard_enabled`, `heavy_enabled`
* `learning_budget` (`timezone`, `daily_model_calls`, `lite_daily_model_calls`,
  `standard_job_model_calls`, `heavy_job_model_calls`, `max_concurrent_model_calls`,
  `default_job_deadline_seconds`)
* `learning_routes` (`reflector`, `curator`, `task`, `reviewer`), each exactly
  `provider` / `model` / `reasoning`

## Limits

* Text-only instruction transfer; no routing, task-tool, filesystem-execution or
  billing attestation.
* Baseline and candidate share one provider, so this is not model independence.
* Small case sets are descriptive, not statistical.
* Heavy is bounded by design: at most 3 rounds and 8 evaluations per job.
