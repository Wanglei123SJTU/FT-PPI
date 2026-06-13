from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TEXT_COL = "description"
Y_COL = "points"


def clean_wine(input_csv: str | Path) -> pd.DataFrame:
    """Load Wine Reviews, keep text/outcome, deduplicate by description."""
    df = pd.read_csv(input_csv)
    missing = {TEXT_COL, Y_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean = df[[TEXT_COL, Y_COL]].dropna().copy()
    clean = clean.drop_duplicates(subset=[TEXT_COL], keep="first")
    clean[Y_COL] = clean[Y_COL].astype(float)
    clean = clean.reset_index(drop=True)
    clean.insert(0, "sample_id", np.arange(len(clean), dtype=np.int64))
    return clean


def sample_population(
    clean_df: pd.DataFrame,
    population_size: int | None,
    seed: int,
) -> pd.DataFrame:
    """Sample a finite population from the cleaned data."""
    if population_size is None or population_size >= len(clean_df):
        return clean_df.copy()
    if population_size <= 0:
        raise ValueError("population_size must be positive")

    rng = np.random.default_rng(seed)
    ids = rng.choice(clean_df["sample_id"].to_numpy(), size=population_size, replace=False)
    return clean_df[clean_df["sample_id"].isin(ids)].sort_values("sample_id").reset_index(drop=True)


def make_split_roles(
    population: pd.DataFrame,
    budget: int,
    train_size: int,
    validation_size: int,
    seed: int,
    replication_id: int = 0,
) -> pd.DataFrame:
    """Assign each population row to train/validation/correction/unlabeled."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if train_size < 0 or validation_size < 0:
        raise ValueError("train_size and validation_size must be nonnegative")
    correction_size = budget - train_size - validation_size
    if correction_size < 0:
        raise ValueError("budget must be at least train_size + validation_size")
    if budget > len(population):
        raise ValueError("budget cannot exceed population size")

    rng = np.random.default_rng(seed)
    sample_ids = population["sample_id"].to_numpy()
    labeled = rng.choice(sample_ids, size=budget, replace=False)
    train = rng.choice(labeled, size=train_size, replace=False) if train_size else np.array([], dtype=labeled.dtype)
    remaining = np.setdiff1d(labeled, train, assume_unique=False)
    validation = (
        rng.choice(remaining, size=validation_size, replace=False)
        if validation_size
        else np.array([], dtype=labeled.dtype)
    )
    correction = np.setdiff1d(remaining, validation, assume_unique=False)
    if len(correction) != correction_size:
        raise RuntimeError("internal split size mismatch")

    roles = pd.DataFrame({"sample_id": sample_ids, "split_role": "unlabeled"})
    roles.loc[roles["sample_id"].isin(train), "split_role"] = "train"
    roles.loc[roles["sample_id"].isin(validation), "split_role"] = "validation"
    roles.loc[roles["sample_id"].isin(correction), "split_role"] = "correction"
    roles["replication_id"] = replication_id
    roles["budget_B"] = budget
    roles["allocation_s"] = train_size
    roles["seed"] = seed
    assert_split_disjoint(roles)
    return roles


def assert_split_disjoint(roles: pd.DataFrame) -> None:
    """Validate that each sample has exactly one split role."""
    if roles["sample_id"].duplicated().any():
        dupes = roles.loc[roles["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"Duplicate sample_id in roles: {dupes}")
    allowed = {"train", "validation", "correction", "unlabeled"}
    observed = set(roles["split_role"].unique())
    if not observed.issubset(allowed):
        raise ValueError(f"Unknown split roles: {sorted(observed - allowed)}")


def build_artifacts(
    input_csv: str | Path,
    output_dir: str | Path,
    population_size: int | None,
    budget: int,
    train_size: int,
    validation_size: int,
    seed: int,
    replication_id: int = 0,
    population_seed: int | None = None,
    split_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and save population and split-role artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clean = clean_wine(input_csv)
    population = sample_population(clean, population_size, seed if population_seed is None else population_seed)
    roles = make_split_roles(
        population,
        budget,
        train_size,
        validation_size,
        seed if split_seed is None else split_seed,
        replication_id,
    )
    population.to_csv(output / "population.csv", index=False)
    roles.to_csv(output / "split_roles.csv", index=False)
    return population, roles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned Wine population and split-role files.")
    parser.add_argument("--input-csv", default="Code/wine_data.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--population-size", type=int, default=5000)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--validation-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--replication-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    population, roles = build_artifacts(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        population_size=args.population_size,
        budget=args.budget,
        train_size=args.train_size,
        validation_size=args.validation_size,
        seed=args.seed,
        replication_id=args.replication_id,
    )
    print(f"population rows: {len(population)}")
    print(roles["split_role"].value_counts().to_dict())


if __name__ == "__main__":
    main()
