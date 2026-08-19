# Local Claude–Codex loop engineering

The repository now has a local, fail-closed orchestration harness under `tools/loop_orchestrator`. It automates maker → independent verifier → rework or planner transitions without implementing Phase 6C itself.

## State and trust boundaries

Every run receives a random run ID and checkpoint ID and records UTC timestamps, role, exit state, and base commit below ignored `.loop/runs/`. The state machine uses `IDLE`, `BUDGET_CHECK`, `MAKER_RUNNING`, `VERIFYING`, `REWORK`, `PLANNING`, `ACCEPTED`, `BLOCKED`, `BUDGET_STOP`, and `FAILED`.

Claude is the only writable model role. Codex runs with `--sandbox read-only` and `--ask-for-approval never`. A Git status/diff snapshot is taken immediately before and after verification; any change converts the result to FAIL. Planning is called only after PASS. The verifier never receives Claude's summary as evidence.

Dry-run is implicit unless `--execute` is present. Execute mode fails closed when the worktree was already dirty, either provider's period usage is missing, stale, future-dated, out of range, or source-less, hard usage is at least 80%, or a new checkpoint would start at or above the 75% soft threshold. The defaults are one checkpoint, two rework rounds per checkpoint, and three repetitions of an identical normalized failure signature.

Only named tests present in `.loop/config.json` are executable. Each entry is a literal argument array passed to `subprocess` with `shell=False`; unknown names block execution. Every required test must PASS before verifier PASS is accepted. A FAIL result overrides model PASS and drives rework, while TIMEOUT or START_FAILED blocks immediately. Status and output tails are persisted to state and handoff. Process timeout, nonzero exit, and JSON parse failures have distinct error codes. Ctrl+C propagates as interruption, while Python's subprocess handling terminates the active child before the CLI exits; the next enhancement should add an explicit Windows process-group/job-object layer for descendant cleanup.

Planner output carries `claude_prompt` into the validated `.loop/next-goal.json`; chained checkpoints include it in the maker's structured goal context. The checked-in example goal is a non-executable template. Execute mode rejects template or placeholder objectives and code checkpoints with empty acceptance criteria or required tests.

## Operations

Initialize local files with `python -m tools.loop_orchestrator init`, then manually populate both `claude.period_usage_percent` and `codex.period_usage_percent` in `.loop/budget.json`. `doctor` is read-only and reports Python, repository/worktree, CLI versions, and selected adapters. `status` prints the latest durable record.

Doctor uses Python subprocesses—not PATH lookup alone—to execute Claude version, Codex version, and Codex exec help. `codex_executable` may point at an independent installation when WindowsApps aliases reject subprocess launch. If that cannot be made executable, the supported next choices are the official Codex SDK in an isolated developer-tool environment or the App Server; completion must not be claimed while doctor is unhealthy. App Server may also expose `account/rateLimits/read`, but any unavailable rate-limit data remains unknown.

Security invariants: no secrets or auth files are read, no environment dump is logged, no model output is treated as a command, no unrestricted permission flags are used, and no Git publishing/deployment operation exists.
