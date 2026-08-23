# Image AI Studio project quality harness

`tools/project_harness` is the deterministic quality gate used after a loop
checkpoint and before committing product changes. It complements rather than
replaces `tools/loop_orchestrator`:

- the loop orchestrator controls Claude maker, Codex verifier, budget, and
  checkpoint transitions;
- the project harness runs fixed local validation profiles and stores durable
  evidence without calling any AI service.

The harness is standard-library-only. It never evaluates generated commands,
uses argument arrays with `shell=False`, defaults to a no-process dry run, and
retains each real run below ignored `.harness/runs/<run-id>/`.

## Commands

Run from the repository root with the active project Python:

```powershell
python -m tools.project_harness list
python -m tools.project_harness doctor
python -m tools.project_harness run --profile phase6c
python -m tools.project_harness run --profile phase6c --execute
python -m tools.project_harness run --profile full --execute
```

Add `--require-clean` for baseline or CI validation. Do not use that option
while validating expected, uncommitted maker changes.

## Profiles

| Profile | Purpose |
| --- | --- |
| `syntax` | Compile `src`, `scripts`, and `tools`. |
| `orchestrator` | Compile the project and run all tests under `tests/tools`. |
| `phase6c` | Run the focused inference/MainWindow/controller/worker GUI gate. |
| `full` | Compile the project and run the complete pytest suite exactly once. |

Every step receives a unique retained directory containing `temp`, `pycache`,
`stdout.log`, `stderr.log`, and `result.json`. The harness overrides only
`TEMP`, `TMP`, `TMPDIR`, `PYTHONUTF8`, `PYTHONIOENCODING`,
`PYTHONPYCACHEPREFIX`, and `QT_QPA_PLATFORM=offscreen` in a copy of the parent
environment. `manifest.json` and `summary.json` record the Python executable,
base commit, exact argv, timeout, status, exit code, bounded output tails, and
Git dirtiness before and after execution.

## Recommended checkpoint flow

1. Run the loop orchestrator for one checkpoint.
2. Inspect its required-test and independent-verifier evidence.
3. Run `phase6c` through this harness once.
4. Run `full` through this harness once.
5. Review the Git scope and commit manually only after both gates pass.

The harness does not commit, push, deploy, download data, run real inference,
or automatically delete artifacts. GPU and real-dataset E2E workflows remain
explicit manual gates because they have materially different cost and hardware
requirements.
