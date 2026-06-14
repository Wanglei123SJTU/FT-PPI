# Wine LoRA-Var Scaling-Law Pipeline

This repository now uses a clean-slate Wine Reviews experiment pipeline. The
old smoke/tiny/allocation pilot code has been removed. The active experiment is
the single-purpose LoRA-Var scaling-law and ramp-up timing run.

## Active Experiment

- Data: `Code/wine_data.csv`
- Config: `configs/wine_var_scaling_b10000.yaml`
- Entrypoint: `python -m src.experiments.wine_var_scaling_law`
- Slurm array: `slurm/run_wine_var_scaling_b10000.sbatch`
- Hyak task: `hyak_tasks/055_wine_var_scaling_b10000.sh`

Current timing-first defaults:

- `B = 10000`
- `R = 1`
- `V = 1000`
- `E_eval = 0`
- `s_grid = [100, 200, 400, 700, 1000, 1500, 2500, 4000]`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- loss: LoRA-Var only

## Commands

Run one Slurm array cell manually:

```bash
python -m src.experiments.wine_var_scaling_law train-cell \
  --config configs/wine_var_scaling_b10000.yaml \
  --task-index 0
```

Aggregate after all eight cells complete:

```bash
python -m src.experiments.wine_var_scaling_law aggregate \
  --config configs/wine_var_scaling_b10000.yaml
```

Submit through Slurm:

```bash
sbatch slurm/run_wine_var_scaling_b10000.sbatch
```

## Persistent Hyak Runner

Start the persistent runner from Windows:

```powershell
scripts\start_hyak_runner.bat
```

The runner pulls `main` and executes new files under `hyak_tasks/*.sh`. The
current task submits the eight-cell Slurm array, waits for completion, runs the
aggregate step, and validates required outputs under
`artifacts/wine_var_scaling_b10000/`.
