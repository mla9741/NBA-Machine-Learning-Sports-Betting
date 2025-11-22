"""Utility helpers for creating richer model features."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


DERIVED_COLUMNS = {
    "Home_Efficiency": ("PTS", "MIN"),
    "Away_Efficiency": ("PTS.1", "MIN.1"),
    "Home_Assist_To_Turnover": ("AST", "TOV"),
    "Away_Assist_To_Turnover": ("AST.1", "TOV.1"),
    "Home_FreeThrow_Rate": ("FTA", "FGA"),
    "Away_FreeThrow_Rate": ("FTA.1", "FGA.1"),
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a ratio while avoiding divide-by-zero explosions."""

    denominator = denominator.replace({0: 1e-9})
    return numerator / denominator


def _ensure_columns(frame: pd.DataFrame, required: Iterable[str]) -> bool:
    """Check whether all columns are present in the dataframe."""

    return all(col in frame.columns for col in required)


def _resolve_column(frame: pd.DataFrame, column_name: str) -> pd.Series:
    """Return a single Series even when duplicate column names exist."""

    series_or_frame = frame.loc[:, column_name]
    if isinstance(series_or_frame, pd.DataFrame):
        # Pandas returns a DataFrame when the column label is duplicated.  We
        # just take the first occurrence so downstream code that expects a
        # Series keeps working the same way it did before augment_features
        # added derived columns.
        return series_or_frame.iloc[:, 0]
    return series_or_frame


def augment_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived performance indicators that models can train on.

    The existing dataset already contains a robust set of per-team counting
    stats.  This helper creates composite metrics (efficiency, turnover
    avoidance, etc.) without mutating the original frame, keeping backwards
    compatibility for callers that still expect the raw columns.
    """

    frame = frame.copy()

    for derived_name, (numerator, denominator) in DERIVED_COLUMNS.items():
        if _ensure_columns(frame, (numerator, denominator)):
            numerator_series = _resolve_column(frame, numerator)
            denominator_series = _resolve_column(frame, denominator)
            frame[derived_name] = _safe_ratio(numerator_series, denominator_series)

    if _ensure_columns(frame, ("REB", "REB.1")):
        frame["Rebound_Differential"] = frame["REB"] - frame["REB.1"]
    if _ensure_columns(frame, ("PTS", "PTS.1")):
        frame["Scoring_Differential"] = frame["PTS"] - frame["PTS.1"]
    if _ensure_columns(frame, ("Days-Rest-Home", "Days-Rest-Away")):
        frame["Rest_Differential"] = frame["Days-Rest-Home"] - frame["Days-Rest-Away"]

    return frame

