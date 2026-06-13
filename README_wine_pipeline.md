# Minimal Wine FT+PPI Pipeline

This is a fresh MVP pipeline for the Wine Reviews rerun. It only uses
`Code/wine_data.csv` and intentionally ignores `FT PPI Original`.

## Local sanity checks

```bash
python -m pytest tests
python -m src.data.build_wine --output-dir artifacts/local_sanity --population-size 5000 --budget 500 --train-size 100 --validation-size 100
```

## Hyak smoke run

Install the Python dependencies from `requirements.txt` in the Hyak environment,
then run:

```bash
sbatch slurm/smoke_lora.sbatch
```

The smoke job trains MSE and Var LoRA runs for two steps each and writes
prediction parquet files under `artifacts/smoke/`.

## Tiny run

```bash
sbatch slurm/run_tiny.sbatch
```

This runs MSE and Var on the `B=500, s=100, v=100` tiny split and writes a
summary table under `artifacts/tiny/summary/metrics.csv`.

