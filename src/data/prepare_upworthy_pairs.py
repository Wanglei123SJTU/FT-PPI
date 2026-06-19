from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_FILENAMES = [
    "upworthy-archive-exploratory-packages-03.12.2020.csv",
    "upworthy-archive-confirmatory-packages-03.12.2020.csv",
    "upworthy-archive-holdout-packages-03.12.2020.csv",
]

REQUIRED_COLUMNS = [
    "clickability_test_id",
    "eyecatcher_id",
    "created_at",
    "headline",
    "excerpt",
    "lede",
    "share_text",
    "impressions",
    "clicks",
]


def _source_name(path: Path) -> str:
    name = path.name.lower()
    if "exploratory" in name:
        return "exploratory"
    if "confirmatory" in name:
        return "confirmatory"
    if "holdout" in name:
        return "holdout"
    return path.stem


def read_raw_archives(raw_dir: Path, filenames: list[str] | None = None) -> pd.DataFrame:
    filenames = filenames or RAW_FILENAMES
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    for filename in filenames:
        path = raw_dir / filename
        if not path.exists():
            missing.append(path)
            continue
        frame = pd.read_csv(path, low_memory=False)
        missing_cols = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing_cols:
            raise ValueError(f"{path} is missing required columns: {missing_cols}")
        frame = frame.copy()
        frame["source_split"] = _source_name(path)
        frames.append(frame)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing raw Upworthy archive file(s):\n{missing_text}")
    if not frames:
        raise ValueError(f"No raw archive files found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def clean_arm_level(raw: pd.DataFrame) -> pd.DataFrame:
    """Build a LOLA-style arm-level CTR table without adopting LOLA's winner task."""
    df = raw[REQUIRED_COLUMNS + ["source_split"]].copy()
    df = df[df["headline"].notna()]

    df["clicks"] = pd.to_numeric(df["clicks"], errors="coerce")
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")
    df = df.dropna(subset=["clicks", "impressions", "created_at"])
    df = df[(df["impressions"] > 0) & (df["clicks"] >= 0) & (df["clicks"] <= df["impressions"])]

    text_cols = ["excerpt", "lede", "share_text"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    df["ctr"] = df["clicks"] / df["impressions"]
    filtered = (
        df.groupby(["clickability_test_id", "eyecatcher_id"])
        .filter(lambda group: len(group["headline"].unique()) > 1)
        .copy()
    )
    dedup_cols = ["clickability_test_id", "eyecatcher_id", "created_at", "headline", "ctr", "clicks", "impressions"]
    arms = filtered.drop_duplicates(subset=dedup_cols)
    arms = arms.sort_values(["created_at", "clickability_test_id", "eyecatcher_id", "headline"]).reset_index(drop=True)
    arms["test_id"] = arms.groupby(["clickability_test_id", "eyecatcher_id"], sort=True).ngroup().astype(int)
    arms["arm_id"] = arms.groupby("test_id").cumcount().astype(int)
    arms["ctr_smoothed"] = (arms["clicks"] + 0.5) / (arms["impressions"] + 1.0)
    arms["logit_ctr_smoothed"] = np.log(arms["ctr_smoothed"] / (1.0 - arms["ctr_smoothed"]))
    return arms[
        [
            "test_id",
            "arm_id",
            "clickability_test_id",
            "eyecatcher_id",
            "created_at",
            "headline",
            "excerpt",
            "lede",
            "share_text",
            "clicks",
            "impressions",
            "ctr",
            "ctr_smoothed",
            "logit_ctr_smoothed",
            "source_split",
        ]
    ]


def headline_count_distribution(arms: pd.DataFrame) -> pd.DataFrame:
    counts = arms.groupby("test_id").size()
    rows = [
        ("2", int((counts == 2).sum())),
        ("3", int((counts == 3).sum())),
        ("4", int((counts == 4).sum())),
        ("5", int((counts == 5).sum())),
        ("6", int((counts == 6).sum())),
        ("7 or more", int((counts >= 7).sum())),
    ]
    table = pd.DataFrame(rows, columns=["num_headlines_in_one_test", "num_tests"])
    total = table["num_tests"].sum()
    table["pct_tests"] = 100.0 * table["num_tests"] / total if total else 0.0
    return table


def lola_literal_arm_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the LOLA paper's package rows for descriptive Table 3 checks."""
    df = raw.copy()
    df["clicks"] = pd.to_numeric(df["clicks"], errors="coerce")
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce")
    df["ctr"] = df["clicks"] / df["impressions"]
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")
    filtered = (
        df.groupby(["clickability_test_id", "eyecatcher_id"])
        .filter(lambda group: len(group["headline"].unique()) > 1)
        .copy()
    )
    cols = ["clickability_test_id", "eyecatcher_id", "created_at", "headline", "ctr", "clicks", "impressions"]
    literal = filtered[cols].drop_duplicates().copy()
    literal["test_id"] = literal.groupby(["clickability_test_id", "eyecatcher_id"]).ngroup().astype(int)
    return literal


def _split_labels(n: int, h_scale_size: int, target_size: int) -> np.ndarray:
    labels = np.full(n, "unused", dtype=object)
    h_end = min(h_scale_size, n)
    target_end = min(h_end + target_size, n)
    labels[:h_end] = "h_scale"
    labels[h_end:target_end] = "target"
    return labels


def make_one_pair_per_test(
    arms: pd.DataFrame,
    seed: int = 20260618,
    h_scale_size: int = 7000,
    target_size: int = 10000,
) -> pd.DataFrame:
    """Sample one unordered headline pair per test and randomly orient A/B for regression."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for test_id, group in arms.groupby("test_id", sort=True):
        group = group.copy()
        group["headline_clean"] = group["headline"].astype(str).str.strip()
        group = group[group["headline_clean"] != ""]
        group = group.sort_values("arm_id").drop_duplicates(subset=["headline_clean"]).reset_index(drop=True)
        if len(group) < 2:
            continue
        picked = rng.choice(len(group), size=2, replace=False)
        first = group.iloc[int(picked[0])]
        second = group.iloc[int(picked[1])]
        if rng.random() < 0.5:
            first, second = second, first

        y_logit = float(first["logit_ctr_smoothed"] - second["logit_ctr_smoothed"])
        y_ctr = float(first["ctr"] - second["ctr"])
        winner_side = "tie"
        if first["ctr"] > second["ctr"]:
            winner_side = "a"
        elif first["ctr"] < second["ctr"]:
            winner_side = "b"

        rows.append(
            {
                "pair_id": len(rows),
                "test_id": int(test_id),
                "clickability_test_id": first["clickability_test_id"],
                "eyecatcher_id": first["eyecatcher_id"],
                "created_at": first["created_at"],
                "arm_id_a": int(first["arm_id"]),
                "arm_id_b": int(second["arm_id"]),
                "headline_a": first["headline"],
                "headline_b": second["headline"],
                "excerpt_a": first["excerpt"],
                "excerpt_b": second["excerpt"],
                "lede_a": first["lede"],
                "lede_b": second["lede"],
                "share_text_a": first["share_text"],
                "share_text_b": second["share_text"],
                "clicks_a": float(first["clicks"]),
                "clicks_b": float(second["clicks"]),
                "impressions_a": float(first["impressions"]),
                "impressions_b": float(second["impressions"]),
                "ctr_a": float(first["ctr"]),
                "ctr_b": float(second["ctr"]),
                "logit_ctr_a": float(first["logit_ctr_smoothed"]),
                "logit_ctr_b": float(second["logit_ctr_smoothed"]),
                "y_logit_ctr_diff": y_logit,
                "y_ctr_diff": y_ctr,
                "winner_side": winner_side,
            }
        )

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs = pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pairs["pair_id"] = np.arange(len(pairs), dtype=np.int64)
    pairs["split"] = _split_labels(len(pairs), h_scale_size=h_scale_size, target_size=target_size)
    return pairs


def lola_default_test_count(raw: pd.DataFrame) -> int:
    """Count tests under the literal LOLA notebook rule, before headline whitespace cleanup."""
    df = raw[["clickability_test_id", "eyecatcher_id", "headline"]].copy()
    filtered = (
        df.groupby(["clickability_test_id", "eyecatcher_id"])
        .filter(lambda group: group["headline"].nunique(dropna=True) > 1)
        .copy()
    )
    if filtered.empty:
        return 0
    return int(filtered.groupby(["clickability_test_id", "eyecatcher_id"]).ngroup().nunique())


def summarize(raw: pd.DataFrame, arms: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    literal = lola_literal_arm_rows(raw)
    cleaned_tests = int(arms["test_id"].nunique()) if not arms.empty else 0
    return {
        "raw_rows": int(len(raw)),
        "lola_literal_tests": int(literal["test_id"].nunique()) if not literal.empty else 0,
        "lola_literal_packages": int(len(literal)),
        "lola_literal_impressions": float(literal["impressions"].sum()) if not literal.empty else 0.0,
        "lola_literal_clicks": float(literal["clicks"].sum()) if not literal.empty else 0.0,
        "arm_rows": int(len(arms)),
        "tests": cleaned_tests,
        "arm_rows_removed_for_missing_headline": int(len(literal) - len(arms)),
        "pairs_removed_by_headline_strip": int(cleaned_tests - len(pairs)),
        "pairs_one_per_test": int(len(pairs)),
        "split_counts": pairs["split"].value_counts().sort_index().to_dict() if "split" in pairs else {},
        "raw_impressions": float(pd.to_numeric(raw["impressions"], errors="coerce").sum()),
        "raw_clicks": float(pd.to_numeric(raw["clicks"], errors="coerce").sum()),
        "arm_impressions": float(arms["impressions"].sum()) if not arms.empty else 0.0,
        "arm_clicks": float(arms["clicks"].sum()) if not arms.empty else 0.0,
        "mean_abs_y_logit_ctr_diff": float(pairs["y_logit_ctr_diff"].abs().mean()) if not pairs.empty else None,
        "notes": [
            "Arm-level cleaning follows the LOLA convention of grouping by clickability_test_id and eyecatcher_id.",
            "The descriptive LOLA Table 3 distribution is computed from literal LOLA package rows, including one missing-headline row.",
            "The experiment arm-level table drops rows without usable headline text.",
            "Pair-level data is adapted for linear M-estimation, not LOLA winner classification.",
            "Each pair-level test contributes one randomly oriented pair by default to avoid within-test dependence in the main analysis.",
            "For pair construction only, headline whitespace is stripped before requiring two distinct headline texts.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Upworthy headline-pair data for FT+PPI M-estimation.")
    parser.add_argument("--raw-dir", type=Path, default=Path("artifacts/download_probe/upworthy_archive"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/upworthy_m_estimation/simple_lola_like"))
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--h-scale-size", type=int, default=7000)
    parser.add_argument("--target-size", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_raw_archives(args.raw_dir)
    arms = clean_arm_level(raw)
    pairs = make_one_pair_per_test(
        arms,
        seed=args.seed,
        h_scale_size=args.h_scale_size,
        target_size=args.target_size,
    )

    arms.to_csv(args.output_dir / "ctr_arms_lola_like.csv", index=False)
    headline_count_distribution(lola_literal_arm_rows(raw)).to_csv(
        args.output_dir / "headline_count_distribution_lola_table3.csv",
        index=False,
    )
    pairs.to_csv(args.output_dir / "pairs_one_per_test.csv", index=False)
    summary = summarize(raw, arms, pairs)
    summary["raw_dir"] = str(args.raw_dir)
    summary["output_dir"] = str(args.output_dir)
    summary["seed"] = int(args.seed)
    summary["h_scale_size_requested"] = int(args.h_scale_size)
    summary["target_size_requested"] = int(args.target_size)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
