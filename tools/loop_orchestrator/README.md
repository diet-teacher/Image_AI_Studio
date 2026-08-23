# Local loop orchestrator

This standard-library-only tool coordinates a writable Claude maker with a read-only Codex verifier and a PASS-only Codex planner. The default is always a dry run; only `--execute` can start AI processes. It never executes commands emitted by a model. Tests are selected by symbolic name from the exact argument arrays in `.loop/config.json`.

## Setup and commands

Run from the repository root (on Windows, use the Python executable available in your development environment):

```powershell
python -m tools.loop_orchestrator doctor
python -m tools.loop_orchestrator init
python -m tools.loop_orchestrator status
python -m tools.loop_orchestrator run --goal tools/loop_orchestrator/example.goal.json --max-checkpoints 1 --dry-run
```

Before execution, edit `.loop/budget.json` and enter the current subscription-period usage percentages for both providers plus an `updated_at` timestamp. Unknown usage blocks paid calls. Existing dirty worktrees also block execution. `example.goal.json` is intentionally marked `"template": true`; copy it to a local goal file, remove that marker, replace the objective, and provide non-empty acceptance criteria and required tests. The template itself can never execute. Once the goal, allowlisted tests, permissions, and budgets have been reviewed, the explicit execution form is:

```powershell
python -m tools.loop_orchestrator run --goal .loop/phase6c-checkpoint-1.json --max-checkpoints 1 --execute
```

Do not add unrestricted permission flags. The generated runtime state and per-run records live in ignored `.loop/`.

Execute mode also performs a fail-closed, non-model preflight before the maker: configured Claude `--version`, Codex `--version`, and Codex's production global approval/read-only/root prefix followed by `exec --help`. Start failure, timeout, or nonzero exit blocks the run and preserves bounded diagnostics in state, the run directory, and handoff. Dry-run skips this probe and continues to invoke no external model or paid API.

## Integration choice

Claude Code exposes non-interactive `-p`, JSON Schema output, UUID session/resume, and `--max-budget-usd`. Session ID, cost, and turn count are read exclusively from the CLI's top-level result envelope. Bash is explicitly denied; Claude receives only Read/Edit/Write/Glob/Grep tools, while required tests run through fixed orchestrator argv. Codex CLI exposes `codex exec --json`, `--output-schema`, `--ephemeral`, and `--sandbox read-only`; it is used for verifier and planner, so no SDK dependency is installed. The verifier prompt contains only goal, base commit, changed files, actual diff, required-test results, and verification artifacts—not the maker's success narrative.

`goal` is a validated JSON object. Absolute paths, traversal, and ambiguous paths are rejected, and any changed file outside `allowed_files` blocks the run. `required_tests` must all resolve to configured allowlist entries regardless of Claude's claimed `tests_run`. After PASS, the validated planner result is saved as `.loop/next-goal.json`. `max_checkpoints` controls accepted checkpoint chaining; `max_rework_rounds` independently controls FAIL corrections within one checkpoint.

Every required test must report PASS before a verifier PASS can be accepted. A test FAIL overrides a verifier PASS and enters normal rework; exhaustion ends in FAILED. TIMEOUT and START_FAILED are infrastructure blocks. Test status and stdout/stderr tails are retained in state and handoff records. Planner `claude_prompt` is preserved in `next-goal.json` and included in the next maker prompt. `--max-checkpoints` must be at least one.

Each required-test attempt keeps its literal allowlist argv and receives a unique, retained `.loop/test-temp/<run-id>/<attempt>-<test>/` directory through `TEMP`, `TMP`, and `TMPDIR`, plus UTF-8 Python environment overrides. Its symbolic name, status, exit code, bounded output tails, argv, and temp path are saved in a `tests` run record before verifier budget checking or invocation.

Process failures carry a stable `kind`: `TIMEOUT`, `PROCESS_START_FAILED`, `NONZERO_EXIT`, or `JSON_PARSE_FAILED`. Start failures and timeouts from maker, verifier, or planner are infrastructure blocks with handoff; ordinary nonzero and JSON parse failures retain FAILED semantics.

The CLI JSON token/cost events may be retained in role logs by a future event-normalization extension, but per-call usage is not treated as subscription-period utilization. Codex App Server rate-limit auto-query is not enabled in this initial implementation; manual period percentages remain authoritative.

## Limits

- Git worktree comparison detects verifier mutation but cannot provide OS-enforced read-only protection beyond Codex's read-only sandbox.
- Subscription usage is not reliably exposed by the selected CLIs, so execution requires 0–100 values with non-future, timezone-aware `updated_at`, a non-empty `source`, and age within `budget_validity_hours`.
- Set `codex_executable` to an independently installed CLI path if a WindowsApps alias cannot be launched by Python. Doctor executes `codex --version` and `codex exec --help` and fails nonzero on access errors. Official Codex SDK or App Server is the documented fallback if no independently executable CLI is available.
- No commit, push, merge, deployment, authentication-file access, or full-environment logging is implemented.
