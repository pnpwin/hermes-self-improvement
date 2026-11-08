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
6. A delta is small and typed: at most 8 operations, 3 files, 8000 characters,
   bound to the base digest it was produced from.
7. A publish needs an `auto_publish` decision *and* a native writer. Without the
   writer the cycle stops at `human_approval_required`; nothing is written.
   `learning_writer` binds the writer to the host's own `skill_manage`, so every
   write keeps the native guards, lock and approval gate. It refuses on base-digest
   drift, checks every patch match and removal digest *before* the first write, and
   reports a staged or refused write as `published: false` — an approval prompt is
   not a publication. The host tool exists in the host, so the lane still needs
   `learning_budget.auto_publish_enabled: true` (default `false`) before it will
   use it: a writer that exists is not permission to write.
8. Heavy adds one bounded reviewer call. `changes_required` blocks publication
   but does not rewrite the gate decision; the finding is preserved as evidence.
9. A lesson never becomes an instruction without a Standard cycle and paired
   evidence.
10. Budgets are local to the profile and are never described as provider quota.
11. `--dedupe` skips a Lite cycle whose evidence already produced a lesson in the
   same scope; without the flag every cycle is charged, because a repeated task
   is a new observation.
12. `report` groups repeated observations (`count >= 2`) as a signal that a
    paired cycle is due; a group is never an automatic instruction change.

## Telemetry scope: why shadow collection is not wired

`ExposureStore` is fed by explicit `exposure` events. Automatic shadow collection
was checked against the installed host and is **not available** through public
surfaces:

* `tools/skills_tool.py` records nothing on `skill_view`; no host hook carries the
  viewed skill's identity.
* Counters live in the sidecar `~/.hermes/skills/.usage.json`
  (`tools/skill_usage.py`), keyed by skill name with `view_count`, `use_count`,
  `patch_count` and timestamps.
* A name is not a revision: the sidecar holds no package digest, so an outcome
  cannot be attributed to the version that produced it.

Attributing outcomes to a revision needs a separate minimal API extension that
emits identity plus digest. Monkey-patching the host is refused, and a counter is
never reported as causal evidence. Until that extension exists the lane stays
explicit-in, explicit-bind, and `unknown` is never reported as `failed`.

## One deliberate deviation from the upstream tool surface

Upstream `AGENTS.md` fixes the plugin at four tools and forbids adding new lanes.
The learning lane adds a fifth, `self_improvement_learning`, because Skillcraft
must dispatch a learning stage with arguments the ledger can check (`action`,
`allow_model_calls`, `cycle_id`, `dry_run`, `evidence`). Calling the plugin through
a shell subprocess instead would move the same authority into an unverifiable
string and lose the per-call reservation.

It is not a second apply/approve/rollback lane: the tool has no verb that mutates
a package on its own, and publication still needs the gate plus the host's native
writer. Reversing the deviation is one line — drop the entry from
`provides_tools` and the handler map; `hermes self-improvement learning` keeps
working, at the cost of a less checkable Skillcraft call path.

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
