# New Dataset Handoff

This note is a lightweight handoff for starting a new dataset or task in this
repository. It intentionally avoids committing to a new experimental design.
Use it to start the next Codex thread with clean context while preserving the
working Hyak and Wine Reviews infrastructure.

## Current Repository State

- The Wine Reviews implementation is the reference implementation for the
  current FT+PPI workflow.
- The stable operating guide is `AGENTS.md`; read it before editing code or
  submitting Hyak jobs.
- Wine data now lives under `Data/`.
- Paper-facing figures for the Wine Reviews section are copied under
  `Figures/Wine_Reviews_new/` so the section can be moved to Overleaf without
  copying the full experiment artifact tree.
- Hyak automation is driven by `scripts/hyak_runner.sh`,
  `scripts/start_hyak_runner.*`, Slurm scripts under `slurm/`, and task files
  under `hyak_tasks/`.
- Experiment outputs under `artifacts/` are local/remote run products and are
  not meant to be the source of truth for future code changes.

## What To Reuse

- The persistent Hyak runner workflow.
- The pattern of one explicit config per experiment.
- The pattern of one Slurm script per experiment family.
- The convention that expensive reruns use new Hyak task filenames.
- The plotting workflow that exports paper-ready figures into a stable
  `Figures/...` folder.
- The data-split discipline: write down exactly which labels are used for
  training, validation, allocation selection, correction, and reporting.

## What Not To Reuse Blindly

- Wine-specific label ranges, scaling constants, budgets, allocation grids, and
  text preprocessing.
- Wine-specific claims in `ft_ppi_experiment_section.tex`.
- Any baseline or metric that depends on Wine rating semantics.
- Any temporary prompting pilot code unless the new task explicitly needs it.

## Before Designing The New Task

Decide these items explicitly before running expensive jobs:

1. What is the finite population?
2. What is the target estimand?
3. What is the label, and does it need scaling?
4. What is the text/input representation?
5. Which labels are allowed for model training, early stopping, allocation
   selection, PPI correction, and final reporting?
6. Which metric validates surrogate quality?
7. Which metric validates inference quality?
8. Which baselines are necessary and affordable?
9. What is the first smoke test that can fail cheaply?
10. What output tables and figures should be produced automatically?

## Suggested Prompt For A New Codex Thread

```text
We are starting a new dataset/task in the same FT-PPI repository.

Please first read AGENTS.md and docs/new_dataset_handoff.md.
The Wine Reviews experiment is the reference implementation, but do not inherit
Wine-specific assumptions unless explicitly stated.

The new dataset/task is not fully designed yet. First inspect the repository and
summarize which parts are reusable, which parts are Wine-specific, and what
decisions we need before implementing the new experiment. Do not run Hyak jobs
or make major code changes until the new experiment design is written down.
```

## Recommended First Steps In The New Thread

1. Inspect `AGENTS.md`, this handoff, and the current `src/experiments/` files.
2. Write a short design memo for the new dataset before coding.
3. Add a new experiment entrypoint instead of overwriting the Wine experiment.
4. Run a local data sanity check before submitting any Hyak job.
5. Start with a smoke-scale Hyak run, then estimate runtime before scaling up.

