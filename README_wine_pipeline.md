# Minimal Wine FT+PPI Pipeline

This is a fresh MVP pipeline for the Wine Reviews rerun. It only uses
`Code/wine_data.csv` and intentionally ignores `FT PPI Original`.

## Local sanity checks

```bash
python -m pytest tests
python -m src.data.build_wine --output-dir artifacts/local_sanity --population-size 5000 --budget 500 --train-size 100 --validation-size 100
```

## Hyak smoke run

Create the Hyak Python environment once:

```bash
cd ~/FT-PPI
bash scripts/setup_hyak_env.sh
```

Normal logins do not need to reinstall dependencies:

```bash
cd ~/FT-PPI
source .venv-hyak/bin/activate
```

Only rebuild the environment when it is broken or dependencies changed:

```bash
RESET=1 bash scripts/setup_hyak_env.sh
```

Then run:

```bash
sbatch slurm/smoke_lora.sbatch
```

To request a specific checkpoint GPU, override the Slurm directives at submit
time. For example:

```bash
sbatch --partition=ckpt --gres=gpu:h200:1 slurm/smoke_lora.sbatch
sbatch --partition=ckpt --gres=gpu:a100:1 slurm/smoke_lora.sbatch
```

The smoke job trains MSE and Var LoRA runs for two steps each and writes
prediction parquet files under `artifacts/smoke/`.

## Tiny run

```bash
sbatch slurm/run_tiny.sbatch
```

This runs MSE and Var on the `B=500, s=100, v=100` tiny split and writes a
summary table under `artifacts/tiny/summary/metrics.csv`.
