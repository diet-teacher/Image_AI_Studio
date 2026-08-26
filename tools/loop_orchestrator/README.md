# Local loop orchestrator

This standard-library-only tool coordinates a writable Claude maker with a read-only Codex verifier and a PASS-only Codex planner. The default is always a dry run; only `--execute` can start AI processes. It never executes commands emitted by a model. Tests are selected by symbolic name from the exact argument arrays in `.loop/config.json`.

## Setup and commands

Run from the repository root (on Windows, use the Python executable available in your development environment):

```powershell
python -m tools.loop_orchestrator doctor
python -m tools.loop_orchestrator init
python -m tools.loop_orchestrator status
python -m tools.loop_orchestrator run --goal tools/loop_orchestrator/example.goal.json --max-checkpoints 1 --dry-run
python -m tools.loop_orchestrator run-phase --manifest path/to/approved.phase.json
```

Before execution, edit `.loop/budget.json` and enter the current subscription-period usage percentages for both providers plus an `updated_at` timestamp. Unknown usage blocks paid calls. Existing dirty worktrees also block execution. `example.goal.json` is intentionally marked `"template": true`; copy it to a local goal file, remove that marker, replace the objective, and provide non-empty acceptance criteria and required tests. The template itself can never execute. Once the goal, allowlisted tests, permissions, and budgets have been reviewed, the explicit execution form is:

```powershell
python -m tools.loop_orchestrator run --goal .loop/phase6c-checkpoint-1.json --max-checkpoints 1 --execute
```

Do not add unrestricted permission flags. The generated runtime state and per-run records live in ignored `.loop/`.

## Pre-approved Phase runs

`run-phase` extends the same fail-closed engine to an ordered set of goals that has been reviewed in advance. Its default is a static dry-run: it validates the manifest, referenced executable goals, fixed test allowlists, repository-relative paths, and numeric limits without probing executables, running tests, or calling a model. Only `--execute` starts the approved sequence; only `--resume` may continue a durable blocked/failed Phase state. Neither form commits, publishes, or starts another Phase.

A Phase manifest declares `phase_id`, `objective`, ordered `{checkpoint_id, goal}` entries, Phase-wide `allowed_files` and `allowed_tests`, one fixed `final_harness_profile`, positive limits for checkpoints, rework, model calls, reported Claude cost, and elapsed time, plus explicit completion conditions. Referenced goals must be non-template executable goals whose file/test scopes are subsets of the Phase. Goal paths are repository-relative, cannot traverse or resolve outside the repository, and cannot be symlinks. See `example.phase.json` for the data shape; it is illustrative and intentionally does not reference an executable checked-in goal.

Phase execution persists both an immutable Phase-start repository snapshot and a content snapshot immediately before each checkpoint's first maker call. Snapshots record existence/type, SHA-256, symlink targets, and bounded before content used to reconstruct textual diffs. The active checkpoint record and its maker/test/verifier stage, session ID, rework context, attempts, and last observed snapshot are durably saved. Verifier input labels the current-checkpoint delta (used for checkpoint allowlist scope) separately from the cumulative Phase-wide delta (context only); prior approved checkpoint changes therefore cannot become current-checkpoint violations. Rework and resume retain the original checkpoint baseline, and the final guard checks the Phase-wide delta against the manifest union.

Every maker, required-test, verifier, and final-harness boundary also fingerprints HEAD, the staged set, tracked/untracked repository files, and protected ignored local files (`.loop/config.json`, `.loop/budget.json`, `.claude/settings.local.json`, and `.vscode/settings.json`). Runtime artifacts below `.loop/runs`, `.loop/test-temp`, and `.harness` remain excluded. Maker mutations are diagnosed even when the maker returns BLOCKED, raises a process/general exception, or is interrupted; unsafe file, HEAD, staged, or protected-local mutations take precedence and block. Tests, verifier, and harness must leave the guarded worktree unchanged.

Manual provider-period usage remains authoritative and must be fresh, timezone-aware, sourced, and within the configured `soft_stop_percent`/`hard_stop_percent` thresholds for both providers; Phase mode defers to `BudgetManager`'s `BudgetDecision` as the single usage policy authority, with no separate Phase-specific percentage. Before every model call the engine rechecks period budget, elapsed time, total calls, and the conservative Claude per-call cost ceiling. Codex cost is not estimated. Phase mode follows the manifest directly and does not call the planner, preventing generated goals from expanding the approved order or scope.

After every verifier passes, and only then, the engine invokes the manifest's named profile from the existing deterministic `project_harness`. Its fixed Python argv arrays, `shell=False`, UTF-8 environment, retained isolated temporary directories, and bounded diagnostics remain authoritative. A passing profile yields `READY_TO_COMMIT`; ordinary failure yields `FAILED`; start failure or timeout yields `BLOCKED`.

Resume is explicit and fail-closed. It requires a genuine Phase record, the same manifest digest and base commit, a complete active-checkpoint baseline/stage record, an unchanged last-observed snapshot, no staged/protected/external changes, fresh budgets, and remaining model/cost/time limits. An incomplete checkpoint is resumed against its original baseline, so pre-timeout maker edits remain in the verifier delta even when the resumed maker makes no edit. Missing baseline data blocks with `RESUME_INCOMPLETE_CHECKPOINT_STATE`; completed checkpoints are never replayed.

The final harness is guarded with the same before/after snapshot. A PASS report cannot produce `READY_TO_COMMIT` if product files, HEAD, staged state, or protected local files changed. Harness startup/runtime exceptions, timeout-style infrastructure reports, and Ctrl+C become structured BLOCKED handoffs rather than escaping with a traceback.

Execute mode also performs a fail-closed, non-model preflight before the maker: configured Claude `--version`, Codex `--version`, and Codex's production global approval/read-only/root prefix followed by `exec --help`. Start failure, timeout, or nonzero exit blocks the run and preserves bounded diagnostics in state, the run directory, and handoff. Dry-run skips this probe and continues to invoke no external model or paid API.

## Integration choice

Claude Code exposes non-interactive `-p`, JSON Schema output, UUID session/resume, and `--max-budget-usd`. Optional `claude_model` configuration adds one explicit `--model`; omission preserves the CLI default. Session, cost, turns, terminal reason, subtype, and validated per-model `modelUsage` (including helper models) are read from the CLI envelope. The primary canonical model is recorded only when an explicitly requested model matches an observed canonical model. Bash is explicitly denied; Claude receives only Read/Edit/Write/Glob/Grep tools, while required tests run through fixed orchestrator argv. Codex CLI exposes `codex exec --json`, `--output-schema`, `--ephemeral`, and `--sandbox read-only`; it is used for verifier and planner, so no SDK dependency is installed.

`goal` is a validated JSON object. Absolute paths, traversal, and ambiguous paths are rejected, and any changed file outside `allowed_files` blocks the run. `required_tests` must all resolve to configured allowlist entries regardless of Claude's claimed `tests_run`. After PASS, the validated planner result is saved as `.loop/next-goal.json`. `max_checkpoints` controls accepted checkpoint chaining; `max_rework_rounds` independently controls FAIL corrections within one checkpoint.

Every required test must report PASS before a verifier PASS can be accepted. A test FAIL overrides a verifier PASS and enters normal rework; exhaustion ends in FAILED. TIMEOUT and START_FAILED are infrastructure blocks. Test status and stdout/stderr tails are retained in state and handoff records. Planner `claude_prompt` is preserved in `next-goal.json` and included in the next maker prompt. `--max-checkpoints` must be at least one.

Each required-test attempt keeps its literal allowlist argv and receives a unique, retained `.loop/test-temp/<run-id>/<attempt>-<test>/` directory through `TEMP`, `TMP`, and `TMPDIR`, plus UTF-8 Python environment overrides. Its symbolic name, status, exit code, bounded output tails, argv, and temp path are saved in a `tests` run record before verifier budget checking or invocation.

Process failures carry a stable `kind`: `PROCESS_START_FAILED`, `PROCESS_TIMEOUT` (aliased as `TIMEOUT` for compatibility), `NONZERO_EXIT`, `JSON_PARSE_FAILED`, `MODEL_BUDGET_EXHAUSTED`, `MODEL_MAX_TURNS`, `API_CONNECTION_ERROR`, or `MAKER_SAFETY_VIOLATION`. Start failures and timeouts from maker, verifier, or planner are infrastructure blocks with handoff; ordinary nonzero and JSON parse failures retain FAILED semantics.

Claude and Codex prompts are sent only over stdin, never as an argv element; argv never contains prompt text, so it cannot appear in a logged command line. On a nonzero Claude exit, the CLI's JSON envelope is parsed if present and bounded, secret-free telemetry (`session_id`, `total_cost_usd`, `num_turns`, `terminal_reason`, `subtype`, `stop_reason`, `errors`, `is_error`, a truncated `result`) is attached to `ProcessFailure.metadata` and to every state/run/handoff record derived from it. Explicit envelope markers classify `MODEL_BUDGET_EXHAUSTED`, `MODEL_MAX_TURNS`, and `API_CONNECTION_ERROR`; anything else nonzero, including malformed nonzero stdout, stays `NONZERO_EXIT`. Codex's json-lines nonzero path is unaffected.

In Phase mode, a reported nonzero-exit cost is still accumulated, and a `MODEL_BUDGET_EXHAUSTED`/`MODEL_MAX_TURNS` failure may resume the same Claude session once per checkpoint with a concise finish-only prompt if a valid session ID exists and every budget/guard check still passes; `API_CONNECTION_ERROR` may retry once in a fresh session only if the failed call left the repository, protected files, HEAD, and the staged set untouched. Both limits default to one and are configurable via `max_maker_continuations` and `max_api_connection_retries`. Safety violations, protected/HEAD/staged changes, `JSON_PARSE_FAILED`, ordinary `NONZERO_EXIT`, and exhausted cost/time/call budgets never retry or continue.

The CLI JSON token/cost events may be retained in role logs by a future event-normalization extension, but per-call usage is not treated as subscription-period utilization. Codex App Server rate-limit auto-query is not enabled in this initial implementation; manual period percentages remain authoritative.

## Limits

- Git worktree comparison detects verifier mutation but cannot provide OS-enforced read-only protection beyond Codex's read-only sandbox.
- Subscription usage is not reliably exposed by the selected CLIs, so execution requires 0–100 values with non-future, timezone-aware `updated_at`, a non-empty `source`, and age within `budget_validity_hours`.
- Set `codex_executable` to an independently installed CLI path if a WindowsApps alias cannot be launched by Python. Doctor executes `codex --version` and `codex exec --help` and fails nonzero on access errors. Official Codex SDK or App Server is the documented fallback if no independently executable CLI is available.
- No commit, push, merge, deployment, authentication-file access, or full-environment logging is implemented.
