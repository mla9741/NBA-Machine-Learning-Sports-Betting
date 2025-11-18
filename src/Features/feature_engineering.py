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
            frame[derived_name] = _safe_ratio(frame[numerator], frame[denominator])

    if _ensure_columns(frame, ("REB", "REB.1")):
        frame["Rebound_Differential"] = frame["REB"] - frame["REB.1"]
    if _ensure_columns(frame, ("PTS", "PTS.1")):
        frame["Scoring_Differential"] = frame["PTS"] - frame["PTS.1"]
    if _ensure_columns(frame, ("Days-Rest-Home", "Days-Rest-Away")):
        frame["Rest_Differential"] = frame["Days-Rest-Home"] - frame["Days-Rest-Away"]

    return frame

