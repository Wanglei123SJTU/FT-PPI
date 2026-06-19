# AGENTS.md

This file records stable working habits for future Codex runs. Keep it focused on operating conventions, Hyak workflow, repository hygiene, and reporting discipline. Do not put experimental designs, methods, baselines, budgets, models, datasets, allocations, or statistical conclusions here; those belong in separate plan/config/report files or user instructions.

## Project Scope

- Treat this repository as the active implementation workspace. Ignore legacy folders unless the user explicitly asks to inspect them.
- The active code is organized under `src/`, with configs in `configs/`, Slurm scripts in `slurm/`, Hyak runner tasks in `hyak_tasks/`, and local/remote outputs under `artifacts/`.
- Do not hard-code credentials, passwords, Duo passcodes, tokens, or private paths into scripts or docs.

## Local Development

- Work from the repository root on Windows: `C:\Users\27497\Desktop\FT PPI Revision`.
- Prefer `rg` for searches and `pytest tests -q` for local verification.
- Use `apply_patch` for manual source edits.
- Keep generated experiment outputs out of commits unless the user explicitly asks to version them.
- Before finalizing code changes, check `git status --short --branch` and mention untracked or modified files that matter.

## LaTeX Workflow

- When compiling `.tex`, prefer a full two-pass compile.
- This Windows TinyTeX install may not have `ctex` or `xeCJK`. For Chinese reports, `xelatex` with `fontspec`, `Microsoft YaHei`, and XeTeX line breaking has worked.
- After every LaTeX compile, clean intermediate files by default. Keep `.tex` and `.pdf`; remove matching `.aux`, `.log`, `.out`, `.toc`, `.fdb_latexmk`, `.fls`, `.synctex.gz`, `.xdv`, `.bbl`, `.blg`, `.bcf`, `.run.xml`, `.nav`, `.snm`, `.vrb`, `.lof`, `.lot`, `.lol`, `.idx`, `.ilg`, and `.ind`.
- Do not delete experiment logs or Hyak logs when cleaning TeX intermediates.

## Hyak Workflow

The stable workflow is a persistent Hyak runner. The user logs in once and completes Duo; Codex then drives work by committing/pushing task files and reading logs.

### Automation Contract

- Codex should operate Hyak through the runner workflow, not by asking the user to type PowerShell, SSH, `sbatch`, `squeue`, or log-inspection commands manually.
- The user is responsible only for the interactive authentication step: entering the UW password and completing Duo when the runner window asks for it.
- After that first login succeeds, Codex should handle the loop end to end:
  1. edit local code/config/task files;
  2. run local checks where feasible;
  3. commit and push to GitHub;
  4. let the remote Hyak runner pull the new commit;
  5. submit/monitor Slurm jobs through `hyak_tasks/*.sh`;
  6. inspect runner, task, Slurm, and artifact logs;
  7. fix failures with another commit and a new task filename.
- Do not ask the user to open a new shell or run Hyak commands unless authentication has expired or a GUI/credential prompt is unavoidable.
- If the runner disconnects or authentication expires, Codex should reopen or restart the runner workflow and ask the user only to complete password/Duo again. After authentication, Codex resumes automation.
- Do not paste or store passwords, Duo passcodes, SSH keys, GitHub tokens, or API keys in scripts, docs, commits, or logs.

### New Conversation Quickstart

A new Codex conversation should be able to use Hyak immediately from this repository:

1. Read this file first.
2. Check whether local code is already pushed:

```powershell
git status --short --branch
```

3. Start or reconnect the persistent runner from Windows:

```powershell
scripts\start_hyak_runner.bat
```

4. Ask the user to enter the UW password and complete Duo in the runner window. Do not ask the user to type any other PowerShell or SSH commands.
5. Once authentication succeeds, drive all Hyak work by editing local files, committing, and pushing to `origin/main`.
6. For every new remote action, create a new `hyak_tasks/<id>_<description>.sh` file, commit it, and push it. The runner will pull and execute it.
7. Monitor local runner output at `artifacts/hyak/hyak_runner.log`. If the local stream is stale or disconnected, inspect the remote runner/task logs by creating a small Hyak task or reconnecting the runner rather than asking the user to manually inspect files.
8. If a task fails, fix code/config locally, create a new task filename, commit, push, and let the runner retry. Do not edit or rerun old task IDs unless intentionally clearing remote runner state.

Minimal Hyak task skeleton:

```bash
#!/bin/bash
set -euo pipefail

echo "example_task_start $(date)"

cd ~/FT-PPI
git status --short --branch

# Submit or run the intended remote command here.
# For Slurm jobs, print sbatch command, job id, squeue polling, sacct final state,
# Slurm log tail, and required output checks.

echo "example_task_done $(date)"
```

### Starting the Runner

- Start from Windows with:

```powershell
scripts\start_hyak_runner.bat
```

- The PowerShell launcher defaults to `lei0603@klone.hyak.uw.edu`, remote repo `~/FT-PPI`, branch `main`, and writes the local stream to `artifacts/hyak/hyak_runner.log`.
- The current launcher starts a detached remote runner by default. After login, the remote runner continues polling even if the local tail window disconnects.
- If interactive foreground behavior is needed, use `scripts\start_hyak_runner.ps1 -Foreground`.

### Runner Behavior

- Remote script: `scripts/hyak_runner.sh`.
- Detached launcher: `scripts/start_hyak_runner_remote.sh`.
- Remote state directory: `.hyak_runner/`.
- Remote runner stream: `.hyak_runner/runner.out`.
- Per-task remote logs: `.hyak_runner/logs/`.
- Task state markers:
  - `.hyak_runner/done/<task_id>`
  - `.hyak_runner/failed/<task_id>`
  - `.hyak_runner/running/<task_id>`

The runner repeatedly:

1. Pulls `origin/main` with fast-forward only.
2. Ensures `.venv-hyak` exists, running `scripts/setup_hyak_env.sh` only when needed.
3. Finds new `hyak_tasks/*.sh` files sorted by filename.
4. Skips task IDs already present in `.hyak_runner/done` or `.hyak_runner/failed`.
5. Runs each new task and tees output to the task log.

Because task identity is the filename without `.sh`, reruns must use a new filename such as `055_descriptive_name.sh`; do not reuse an old task name unless you intentionally remove remote state markers.

## Hyak Task Conventions

Each `hyak_tasks/*.sh` should:

- Start with `#!/bin/bash` and `set -euo pipefail` unless there is a specific reason not to.
- Print a clear `*_task_start` marker and a final `*_task_done` marker.
- Assume it is run from `~/FT-PPI` after the runner has pulled `main`.
- Submit Slurm jobs with explicit, logged `sbatch` commands.
- Prefer H200 when useful, but include practical fallback attempts when queue or partition availability is uncertain. Completion is more important than using the most powerful GPU.
- Capture and print the Slurm job ID.
- Poll `squeue` until the job exits.
- Print `sacct` final status and exit code.
- Tail the relevant Slurm log.
- Validate required output files and print concise summaries.
- Never delete old artifacts just to make a rerun clean; use a new output directory or a new task ID.

If a task discovers that a previous job actually completed after a runner disconnect, it should recover by reading existing artifacts and printing a complete summary rather than blindly rerunning expensive work.

## Slurm and Environment Notes

- Hyak jobs should activate `.venv-hyak`; the setup script places the environment and caches in scratch/group storage when available.
- Do not reinstall the Python environment on every login. Use `scripts/setup_hyak_env.sh` only when `.venv-hyak` is missing or dependencies changed. Use `RESET=1 bash scripts/setup_hyak_env.sh` only for a deliberate rebuild.
- Avoid writing large unnecessary artifacts. Hyak storage and quota pressure can become a bottleneck; save only the outputs needed for verification or follow-up work.
- Slurm scripts should write logs to a predictable `logs/` path and artifacts to a predictable `artifacts/<run_name>/` path.
- If the local runner tail disconnects, do not assume the job failed. Reconnect or inspect remote `.hyak_runner/runner.out`, `.hyak_runner/logs/`, Slurm logs, `squeue`, and `sacct`.
- Before submitting GPU jobs, first inspect currently idle GPU resources and choose the best idle type rather than waiting indefinitely for one preferred model. Use `scripts/choose_hyak_gpu.sh` when possible. The default priority is `H200 > A100 > L40S > L40 > A40 > RTX6000 > 2080Ti/P100`; use a bare `--gres=gpu:1` only as a last-resort fallback.
- For array-style experiments, request as many suitable idle GPUs as is practical so independent cells can run in parallel. Prefer a single homogeneous GPU type for a controlled comparison when enough devices are idle. If Slurm runs cells on different suitable GPU types, the experiment must still use the same committed config, seeds, model, optimizer, batch policy, stopping rule, and evaluation protocol across all methods; do not let GPU-specific adjustments create an unfair comparison.

## GitHub and Runner Interaction

- The Hyak runner only sees changes after they are committed and pushed to `origin/main`.
- For Hyak work, the normal loop is:
  1. Edit code/config/task locally.
  2. Run local tests where feasible.
  3. Commit and push to `main`.
  4. Let the runner pull and execute the new task.
  5. Monitor `artifacts/hyak/hyak_runner.log` and remote task/Slurm logs.
  6. Fix with a new commit and a new task filename if needed.
- Do not create broad, unrelated commits. Keep task-specific fixes scoped.

### Standard GitHub Push Procedure

When the user asks to save or push work to GitHub, Codex should do the Git operations directly:

1. Check the worktree:

```powershell
git status --short --branch
```

2. Inspect the intended changes with `git diff --stat`, `git diff --name-status`, and targeted diffs for sensitive files.
3. Run relevant checks, usually:

```powershell
$env:PYTHONPATH='.'; pytest tests -q
```

4. Stage only the intended files. Do not stage local scratch files, credentials, large raw artifacts, or unrelated generated outputs.
5. Commit with a concise message:

```powershell
git commit -m "Concise message"
```

6. Push to GitHub:

```powershell
git push origin main
```

7. Confirm the result:

```powershell
git status --short --branch
git log -1 --oneline
```

If `git push` is rejected because the remote moved, run `git pull --ff-only` and inspect the result before retrying. Do not use force push, reset, or rebase unless the user explicitly asks.

## Reporting Results

- Treat results as verified only when there is a completed task marker or a Slurm `COMPLETED` status with expected output files.
- If the runner disconnects before a final summary is printed, mark the task as submitted or partially observed, not completed.
- Prefer reporting exact artifact paths, task IDs, Slurm job IDs, config names, and key metrics.
- For long experiments, after the first few cells/replications complete, estimate runtime from the observed per-cell/per-replication durations and report an approximate remaining time and total wall-clock expectation.
- After an experiment finishes, proactively inspect the outputs, make clear and readable visualizations for the central metrics when possible, and report the most important empirical takeaways without waiting for the user to ask.
- Distinguish clearly between:
  - pipeline smoke success,
  - engineering viability,
  - pilot evidence,
  - final statistical conclusions.

## Scope Boundary

- Keep experiment design out of this file. If a plan changes methods, budgets, models, datasets, allocations, metrics, or baselines, update the dedicated design document, configs, tasks, or reports instead of this workflow guide.
- When in doubt, use this file only for instructions that should remain true across many experimental redesigns.
