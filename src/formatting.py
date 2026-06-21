"""Small formatting helpers with no optional third-party dependencies."""

from __future__ import annotations

import math
from numbers import Real

import pandas as pd


def _format_cell(value: object, floatfmt: str) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return format(numeric, floatfmt)
        return str(value)
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, *, index: bool = False, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table without tabulate."""
    if index:
        frame = df.reset_index()
    else:
        frame = df.reset_index(drop=True)

    columns = [str(col) for col in frame.columns]
    rows = [[_format_cell(row[col], floatfmt) for col in frame.columns] for _, row in frame.iterrows()]
    widths = [len(col) for col in columns]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(values: list[str]) -> str:
        padded = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    header = render_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row(row) for row in rows]
    return "\n".join([header, separator, *body])
