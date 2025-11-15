"""Feature engineering helpers for model enrichment.

The module centres around :class:`FeatureEngineer` which augments the raw
match level dataframe produced by :mod:`Create_Games` with contextual
signals required by the side and totals models.  The goal is to keep the
existing data pipeline intact while layering additional numeric features
that capture form, travel, fatigue, injury and stylistic tendencies.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from src.DataProviders.BallDontLieClient import BallDontLieClient, DEFAULT_LOOKBACK_DAYS
from src.Utils.geospatial import haversine_distance, to_coordinate
from src.Utils.team_metadata import TeamMetadata, get_team_metadata, TEAM_METADATA

TEAM_STAT_COLUMNS = [
    "PTS",
    "REB",
    "AST",
    "TOV",
    "OREB",
    "DREB",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "PLUS_MINUS",
    "MIN",
]


@dataclass
class TeamStats:
    """Container for per-team statistics derived from a dataframe row."""

    pts: float
    reb: float
    ast: float
    tov: float
    oreb: float
    dreb: float
    fgm: float
    fga: float
    fg_pct: float
    fg3m: float
    fg3a: float
    fg3_pct: float
    ftm: float
    fta: float
    ft_pct: float
    plus_minus: float
    minutes: float

    @property
    def possessions(self) -> float:
        return self.fga + 0.44 * self.fta - self.oreb + self.tov

    @property
    def efg(self) -> float:
        if self.fga == 0:
            return 0.0
        return (self.fgm + 0.5 * self.fg3m) / self.fga

    @property
    def ft_rate(self) -> float:
        if self.fga == 0:
            return 0.0
        return self.ftm / self.fga

    @property
    def tov_rate(self) -> float:
        poss = self.possessions
        if poss == 0:
            return 0.0
        return self.tov / poss

    def orb_rate(self, opponent_dreb: float) -> float:
        denom = self.oreb + opponent_dreb
        if denom == 0:
            return 0.0
        return self.oreb / denom


class FeatureEngineer:
    """Enriches the base matchup dataframe with advanced features."""

    def __init__(
        self,
        injury_client: Optional[BallDontLieClient] = None,
        injury_lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        # Allow feature generation to leverage live Ball Don't Lie data when an
        # API key is configured.  This keeps offline workflows fast (the
        # default client falls back to cached/empty responses) while making it
        # easy to opt into richer injury context.
        if injury_client is None:
            api_key = os.getenv("BALLDONTLIE_API_KEY")
            enable_http = api_key is not None
            injury_client = BallDontLieClient(enable_http=enable_http, api_key=api_key)

        self.injury_client = injury_client
        self.injury_lookback_days = injury_lookback_days

        # Stable numeric identifiers per franchise derived from the metadata
        # catalog.  Aliases are resolved via ``get_team_metadata`` at lookup
        # time.
        ordered_metadata = sorted(TEAM_METADATA.values(), key=lambda meta: meta.name)
        self.team_index: Dict[str, int] = {
            meta.name: idx for idx, meta in enumerate(ordered_metadata)
        }

    # ------------------------------------------------------------------
    def enhance(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        enriched["_Original_Order"] = np.arange(len(enriched))
        enriched["Date"] = pd.to_datetime(enriched["Date"])
        if "Date.1" in enriched:
            enriched["Date.1"] = pd.to_datetime(enriched["Date.1"])

        chronological = enriched.sort_values("Date").reset_index(drop=True)
        chronological["Game_ID"] = chronological.index

        # Numeric identifiers for downstream modelling
        chronological["Home_Team_ID"] = chronological["TEAM_NAME"].apply(
            self._team_numeric_id
        )
        chronological["Away_Team_ID"] = chronological["TEAM_NAME.1"].apply(
            self._team_numeric_id
        )

        schedule = self._build_schedule(chronological)
        schedule = self._compute_schedule_features(schedule)
        schedule = self._apply_travel_features(schedule)
        chronological = self._merge_role_features(chronological, schedule)
        chronological = self._add_location_metadata(chronological)
        chronological = self._add_injury_features(chronological)
        chronological = self._add_matchup_history(chronological)
        chronological = self._add_totals_composites(chronological)

        # Replace potential NaNs with zeros to keep the downstream ML
        # pipelines agnostic to missing data and restore original ordering.
        numeric_cols = chronological.select_dtypes(include=[np.number]).columns
        chronological[numeric_cols] = chronological[numeric_cols].fillna(0.0)
        chronological.sort_values("_Original_Order", inplace=True)
        chronological.drop(columns=["Game_ID", "_Original_Order"], inplace=True)
        chronological.reset_index(drop=True, inplace=True)
        return chronological

    # ------------------------------------------------------------------
    def _team_numeric_id(self, team_name: str) -> int:
        metadata = get_team_metadata(team_name)
        if metadata is None:
            return -1
        return self.team_index.get(metadata.name, -1)

    def _get_team_stats(self, row: pd.Series, suffix: str = "") -> TeamStats:
        values = {}
        for column in TEAM_STAT_COLUMNS:
            key = f"{column}{suffix}"
            try:
                values[column] = float(row.get(key, 0.0))
            except (TypeError, ValueError):
                values[column] = 0.0
        return TeamStats(
            pts=values["PTS"],
            reb=values["REB"],
            ast=values["AST"],
            tov=values["TOV"],
            oreb=values["OREB"],
            dreb=values["DREB"],
            fgm=values["FGM"],
            fga=values["FGA"],
            fg_pct=values["FG_PCT"],
            fg3m=values["FG3M"],
            fg3a=values["FG3A"],
            fg3_pct=values["FG3_PCT"],
            ftm=values["FTM"],
            fta=values["FTA"],
            ft_pct=values["FT_PCT"],
            plus_minus=values["PLUS_MINUS"],
            minutes=values["MIN"],
        )

    def _stats_record(self, team_stats: TeamStats, opponent_stats: TeamStats) -> Dict[str, float]:
        return {
            "Team_PTS": team_stats.pts,
            "Team_REB": team_stats.reb,
            "Team_AST": team_stats.ast,
            "Team_TOV": team_stats.tov,
            "Team_OREB": team_stats.oreb,
            "Team_DREB": team_stats.dreb,
            "Team_FGM": team_stats.fgm,
            "Team_FGA": team_stats.fga,
            "Team_FG3M": team_stats.fg3m,
            "Team_FG3A": team_stats.fg3a,
            "Team_FTM": team_stats.ftm,
            "Team_FTA": team_stats.fta,
            "Team_MIN": team_stats.minutes,
            "Opponent_PTS": opponent_stats.pts,
            "Opponent_REB": opponent_stats.reb,
            "Opponent_TOV": opponent_stats.tov,
            "Opponent_OREB": opponent_stats.oreb,
            "Opponent_DREB": opponent_stats.dreb,
            "Opponent_FGA": opponent_stats.fga,
            "Opponent_FTA": opponent_stats.fta,
            "Opponent_MIN": opponent_stats.minutes,
        }

    def _build_schedule(self, frame: pd.DataFrame) -> pd.DataFrame:
        records: List[Dict] = []
        for _, row in frame.iterrows():
            game_id = row["Game_ID"]
            date = row["Date"]
            home_team = row["TEAM_NAME"]
            away_team = row["TEAM_NAME.1"]
            home_stats = self._get_team_stats(row, suffix="")
            away_stats = self._get_team_stats(row, suffix=".1")

            home_record = {
                "Game_ID": game_id,
                "Team": home_team,
                "Opponent": away_team,
                "Game_Date": date,
                "Is_Home": True,
                "Location_Team": home_team,
            }
            home_record.update(self._stats_record(home_stats, away_stats))

            away_record = {
                "Game_ID": game_id,
                "Team": away_team,
                "Opponent": home_team,
                "Game_Date": date,
                "Is_Home": False,
                "Location_Team": home_team,
            }
            away_record.update(self._stats_record(away_stats, home_stats))

            records.append(home_record)
            records.append(away_record)
        schedule = pd.DataFrame(records)
        schedule.sort_values(["Team", "Game_Date", "Is_Home"], inplace=True)
        schedule.reset_index(drop=True, inplace=True)
        return schedule

    def _compute_schedule_features(self, schedule: pd.DataFrame) -> pd.DataFrame:
        team_possessions = []
        opponent_possessions = []
        pace = []
        off_rating = []
        def_rating = []
        efg = []
        ft_rate = []
        tov_rate = []
        orb_rate = []

        for _, row in schedule.iterrows():
            team_poss = row["Team_FGA"] + 0.44 * row["Team_FTA"] - row["Team_OREB"] + row["Team_TOV"]
            opp_poss = (
                row["Opponent_FGA"] + 0.44 * row["Opponent_FTA"] - row["Opponent_OREB"] + row["Opponent_TOV"]
            )
            minutes = row["Team_MIN"] or 240.0
            minutes = minutes if minutes else 240.0

            current_pace = 48.0 * (team_poss + opp_poss) / (2.0 * (minutes / 5.0))
            current_off = 100.0 * row["Team_PTS"] / team_poss if team_poss else 0.0
            current_def = 100.0 * row["Opponent_PTS"] / opp_poss if opp_poss else 0.0

            team_possessions.append(team_poss)
            opponent_possessions.append(opp_poss)
            pace.append(current_pace)
            off_rating.append(current_off)
            def_rating.append(current_def)

            if row["Team_FGA"]:
                efg.append((row["Team_FGM"] + 0.5 * row["Team_FG3M"]) / row["Team_FGA"])
                ft_rate.append(row["Team_FTM"] / row["Team_FGA"])
            else:
                efg.append(0.0)
                ft_rate.append(0.0)

            tov_rate.append(row["Team_TOV"] / team_poss if team_poss else 0.0)
            denom = row["Team_OREB"] + row["Opponent_DREB"]
            orb_rate.append(row["Team_OREB"] / denom if denom else 0.0)

        schedule = schedule.copy()
        schedule["Team_Possessions"] = team_possessions
        schedule["Opponent_Possessions"] = opponent_possessions
        schedule["Team_Pace"] = pace
        schedule["Team_Off_Rating"] = off_rating
        schedule["Team_Def_Rating"] = def_rating
        schedule["Team_EFG"] = efg
        schedule["Team_FT_Rate"] = ft_rate
        schedule["Team_TOV_Rate"] = tov_rate
        schedule["Team_ORB_Rate"] = orb_rate

        for window in (5, 10):
            schedule[f"Form_PTS_{window}"] = (
                schedule.groupby("Team")["Team_PTS"].apply(
                    lambda series: series.shift(1).rolling(window, min_periods=1).mean()
                )
            )
            schedule[f"Form_DEF_{window}"] = (
                schedule.groupby("Team")["Opponent_PTS"].apply(
                    lambda series: series.shift(1).rolling(window, min_periods=1).mean()
                )
            )
            schedule[f"Form_Pace_{window}"] = (
                schedule.groupby("Team")["Team_Pace"].apply(
                    lambda pace_series: pace_series.shift(1).rolling(window, min_periods=1).mean()
                )
            )
            schedule[f"Form_Off_{window}"] = (
                schedule.groupby("Team")["Team_Off_Rating"].apply(
                    lambda series: series.shift(1).rolling(window, min_periods=1).mean()
                )
            )
            schedule[f"Form_Def_{window}"] = (
                schedule.groupby("Team")["Team_Def_Rating"].apply(
                    lambda series: series.shift(1).rolling(window, min_periods=1).mean()
                )
            )
            schedule[f"Form_EFG_{window}"] = (
                schedule.groupby("Team")["Team_EFG"].apply(
                    lambda series: series.shift(1).rolling(window, min_periods=1).mean()
                )
            )

        return schedule

    def _apply_travel_features(self, schedule: pd.DataFrame) -> pd.DataFrame:
        travel_records: List[Dict] = []
        for team, group in schedule.groupby("Team"):
            group = group.sort_values("Game_Date")
            previous_dates: List[pd.Timestamp] = []
            previous_coord = None
            previous_date: Optional[pd.Timestamp] = None
            home_metadata = get_team_metadata(team)

            for _, row in group.iterrows():
                location_team = row["Location_Team"]
                location_metadata = get_team_metadata(location_team)
                try:
                    current_coord = to_coordinate(location_metadata)
                except ValueError:
                    current_coord = None

                travel_miles = 0.0
                if previous_coord is not None and current_coord is not None:
                    travel_miles = haversine_distance(previous_coord, current_coord)

                days_since_last = (
                    (row["Game_Date"] - previous_date).days if previous_date is not None else None
                )
                b2b = 1.0 if days_since_last == 1 else 0.0
                three_in_four = 0.0
                four_in_six = 0.0
                if previous_dates:
                    recent_four = [
                        d for d in previous_dates if (row["Game_Date"] - d).days <= 3
                    ]
                    recent_six = [
                        d for d in previous_dates if (row["Game_Date"] - d).days <= 5
                    ]
                    if len(recent_four) >= 2:
                        three_in_four = 1.0
                    if len(recent_six) >= 3:
                        four_in_six = 1.0

                altitude = location_metadata.altitude_ft if location_metadata else 0.0
                tz_delta = 0.0
                if location_metadata and home_metadata:
                    tz_delta = self._timezone_delta(home_metadata, location_metadata, row["Game_Date"])

                arena_latitude = current_coord[0] if current_coord else 0.0
                arena_longitude = current_coord[1] if current_coord else 0.0

                travel_records.append(
                    {
                        "Game_ID": row["Game_ID"],
                        "Team": team,
                        "Is_Home": row["Is_Home"],
                        "Travel_Miles": travel_miles,
                        "Current_Altitude": altitude,
                        "Timezone_Delta": tz_delta,
                        "Fatigue_B2B": b2b,
                        "Fatigue_3in4": three_in_four,
                        "Fatigue_4in6": four_in_six,
                        "Days_Since_Last_Game": float(days_since_last) if days_since_last is not None else 7.0,
                        "Arena_Latitude": arena_latitude,
                        "Arena_Longitude": arena_longitude,
                    }
                )

                previous_dates.append(row["Game_Date"])
                previous_coord = current_coord
                previous_date = row["Game_Date"]

        travel_df = pd.DataFrame(travel_records)
        merged = schedule.merge(travel_df, on=["Game_ID", "Team", "Is_Home"], how="left")
        return merged

    def _timezone_delta(
        self,
        base_metadata: Optional[TeamMetadata],
        location_metadata: Optional[TeamMetadata],
        game_date: pd.Timestamp,
    ) -> float:
        if base_metadata is None or location_metadata is None:
            return 0.0
        base_offset = self._timezone_offset(base_metadata.timezone, game_date)
        location_offset = self._timezone_offset(location_metadata.timezone, game_date)
        return location_offset - base_offset

    @lru_cache(maxsize=None)
    def _timezone_offset(self, timezone_name: str, game_date: pd.Timestamp) -> float:
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            return 0.0
        naive_dt = pd.Timestamp(game_date).to_pydatetime().replace(tzinfo=tz)
        offset = naive_dt.utcoffset()
        if offset is None:
            return 0.0
        return offset.total_seconds() / 3600.0

    def _merge_role_features(self, frame: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Team_Pace": "Pace",
            "Team_Off_Rating": "Off_Rating",
            "Team_Def_Rating": "Def_Rating",
            "Team_Possessions": "Possessions",
            "Opponent_Possessions": "Opponent_Possessions",
            "Team_EFG": "EFG",
            "Team_FT_Rate": "FT_Rate",
            "Team_TOV_Rate": "Turnover_Rate",
            "Team_ORB_Rate": "ORB_Rate",
            "Form_PTS_5": "Form_PTS_5",
            "Form_PTS_10": "Form_PTS_10",
            "Form_DEF_5": "Form_DEF_5",
            "Form_DEF_10": "Form_DEF_10",
            "Form_Pace_5": "Form_Pace_5",
            "Form_Pace_10": "Form_Pace_10",
            "Form_Off_5": "Form_Off_5",
            "Form_Off_10": "Form_Off_10",
            "Form_Def_5": "Form_Def_Rating_5",
            "Form_Def_10": "Form_Def_Rating_10",
            "Form_EFG_5": "Form_EFG_5",
            "Form_EFG_10": "Form_EFG_10",
            "Travel_Miles": "Travel_Miles",
            "Current_Altitude": "Current_Altitude",
            "Timezone_Delta": "Timezone_Delta",
            "Fatigue_B2B": "Fatigue_B2B",
            "Fatigue_3in4": "Fatigue_3in4",
            "Fatigue_4in6": "Fatigue_4in6",
            "Days_Since_Last_Game": "Days_Since_Last_Game",
            "Arena_Latitude": "Arena_Latitude",
            "Arena_Longitude": "Arena_Longitude",
        }

        base_cols = ["Game_ID", "Team", "Is_Home"] + list(rename_map.keys())
        role_df = schedule[base_cols].rename(columns=rename_map)
        home_df = (
            role_df[role_df["Is_Home"]]
            .drop(columns=["Is_Home"])
            .set_index("Game_ID")
            .add_prefix("Home_")
        )
        away_df = (
            role_df[~role_df["Is_Home"]]
            .drop(columns=["Is_Home"])
            .set_index("Game_ID")
            .add_prefix("Away_")
        )

        merged = frame.set_index("Game_ID")
        merged = merged.join(home_df, how="left")
        merged = merged.join(away_df, how="left")

        # Pace differentials (opponent adjustments)
        merged["Home_Pace_Adjusted"] = merged["Home_Pace"] - merged["Away_Pace"]
        merged["Away_Pace_Adjusted"] = merged["Away_Pace"] - merged["Home_Pace"]

        merged.reset_index(inplace=True)
        return merged

    def _add_location_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        home_latitudes: List[float] = []
        home_longitudes: List[float] = []
        home_altitudes: List[float] = []
        home_timezones: List[float] = []
        away_latitudes: List[float] = []
        away_longitudes: List[float] = []
        away_altitudes: List[float] = []
        away_timezones: List[float] = []

        for _, row in frame.iterrows():
            date = pd.Timestamp(row["Date"])
            home_meta = get_team_metadata(row["TEAM_NAME"])
            away_meta = get_team_metadata(row["TEAM_NAME.1"])

            if home_meta:
                home_latitudes.append(home_meta.latitude)
                home_longitudes.append(home_meta.longitude)
                home_altitudes.append(home_meta.altitude_ft)
                home_timezones.append(self._timezone_offset(home_meta.timezone, date))
            else:
                home_latitudes.append(0.0)
                home_longitudes.append(0.0)
                home_altitudes.append(0.0)
                home_timezones.append(0.0)

            if away_meta:
                away_latitudes.append(away_meta.latitude)
                away_longitudes.append(away_meta.longitude)
                away_altitudes.append(away_meta.altitude_ft)
                away_timezones.append(self._timezone_offset(away_meta.timezone, date))
            else:
                away_latitudes.append(0.0)
                away_longitudes.append(0.0)
                away_altitudes.append(0.0)
                away_timezones.append(0.0)

        frame = frame.copy()
        frame["Home_Base_Latitude"] = home_latitudes
        frame["Home_Base_Longitude"] = home_longitudes
        frame["Home_Base_Altitude"] = home_altitudes
        frame["Home_Timezone_Offset"] = home_timezones
        frame["Away_Base_Latitude"] = away_latitudes
        frame["Away_Base_Longitude"] = away_longitudes
        frame["Away_Base_Altitude"] = away_altitudes
        frame["Away_Timezone_Offset"] = away_timezones
        frame["Game_Latitude"] = frame.get("Home_Arena_Latitude", frame["Home_Base_Latitude"])
        frame["Game_Longitude"] = frame.get("Home_Arena_Longitude", frame["Home_Base_Longitude"])
        return frame

    def _add_injury_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.injury_client is None:
            frame["Home_Injury_Score"] = 0.0
            frame["Away_Injury_Score"] = 0.0
        else:
            cache: Dict[tuple, float] = {}
            home_scores = []
            away_scores = []
            for _, row in frame.iterrows():
                date = pd.Timestamp(row["Date"]).date()
                home_team = row["TEAM_NAME"]
                away_team = row["TEAM_NAME.1"]
                home_scores.append(self._injury_score(cache, home_team, date))
                away_scores.append(self._injury_score(cache, away_team, date))
            frame["Home_Injury_Score"] = home_scores
            frame["Away_Injury_Score"] = away_scores

        frame["Injury_Score_Diff"] = frame["Home_Injury_Score"] - frame["Away_Injury_Score"]
        frame["Injury_Score_Total_Adjustment"] = frame["Home_Injury_Score"] + frame["Away_Injury_Score"]
        return frame

    def _injury_score(self, cache: Dict[tuple, float], team: str, date: dt.date) -> float:
        key = (team, date)
        if key not in cache:
            cache[key] = self.injury_client.get_team_injury_score(
                team,
                date,
                lookback_days=self.injury_lookback_days,
            )
        return cache[key]

    def _add_matchup_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        matchup_keys = frame.apply(
            lambda row: tuple(sorted([row["TEAM_NAME"], row["TEAM_NAME.1"]])), axis=1
        )
        frame = frame.copy()
        frame["Matchup_Key"] = matchup_keys
        frame["Historical_Total_Avg"] = (
            frame.groupby("Matchup_Key")["Score"].apply(
                lambda series: series.shift(1).rolling(5, min_periods=1).mean()
            )
        )
        frame["Historical_Total_Std"] = (
            frame.groupby("Matchup_Key")["Score"].apply(
                lambda series: series.shift(1).rolling(5, min_periods=1).std()
            ).fillna(0.0)
        )
        frame.drop(columns=["Matchup_Key"], inplace=True)
        return frame

    def _add_totals_composites(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["Game_Pace"] = (frame.get("Home_Pace", 0.0) + frame.get("Away_Pace", 0.0)) / 2.0
        frame["Combined_Possessions"] = (
            frame.get("Home_Possessions", 0.0) + frame.get("Away_Possessions", 0.0)
        )
        frame["Combined_Travel_Fatigue"] = (
            frame.get("Home_Travel_Miles", 0.0) / 1000.0
            + frame.get("Away_Travel_Miles", 0.0) / 1000.0
            + 0.5 * (frame.get("Home_Fatigue_B2B", 0.0) + frame.get("Away_Fatigue_B2B", 0.0))
            + 0.75 * (frame.get("Home_Fatigue_3in4", 0.0) + frame.get("Away_Fatigue_3in4", 0.0))
            + 1.0 * (frame.get("Home_Fatigue_4in6", 0.0) + frame.get("Away_Fatigue_4in6", 0.0))
        )

        # Season level offence/defence anchors (derived from per game stats)
        frame["Home_Season_Offense"] = frame.get("Home_Off_Rating", frame.get("PTS", 0.0))
        frame["Home_Season_Defense"] = frame.get("Home_Def_Rating", frame.get("PTS.1", 0.0))
        frame["Away_Season_Offense"] = frame.get("Away_Off_Rating", frame.get("PTS.1", 0.0))
        frame["Away_Season_Defense"] = frame.get("Away_Def_Rating", frame.get("PTS", 0.0))

        # Form offence/defence proxies
        frame["Home_Form_Offense_5"] = frame.get("Home_Form_Off_5", 0.0)
        frame["Home_Form_Offense_10"] = frame.get("Home_Form_Off_10", 0.0)
        frame["Home_Form_Defense_5"] = frame.get("Home_Form_Def_Rating_5", 0.0)
        frame["Home_Form_Defense_10"] = frame.get("Home_Form_Def_Rating_10", 0.0)
        frame["Away_Form_Offense_5"] = frame.get("Away_Form_Off_5", 0.0)
        frame["Away_Form_Offense_10"] = frame.get("Away_Form_Off_10", 0.0)
        frame["Away_Form_Defense_5"] = frame.get("Away_Form_Def_Rating_5", 0.0)
        frame["Away_Form_Defense_10"] = frame.get("Away_Form_Def_Rating_10", 0.0)

        # Shooting and rate differentials
        frame["Shooting_FG_PCT_Diff"] = frame.get("FG_PCT", 0.0) - frame.get("FG_PCT.1", 0.0)
        frame["Shooting_FG3_PCT_Diff"] = frame.get("FG3_PCT", 0.0) - frame.get("FG3_PCT.1", 0.0)
        frame["Shooting_FT_PCT_Diff"] = frame.get("FT_PCT", 0.0) - frame.get("FT_PCT.1", 0.0)
        frame["Rebound_Diff"] = frame.get("REB", 0.0) - frame.get("REB.1", 0.0)
        frame["Assist_Diff"] = frame.get("AST", 0.0) - frame.get("AST.1", 0.0)
        frame["Turnover_Diff"] = frame.get("TOV", 0.0) - frame.get("TOV.1", 0.0)

        # Estimated rates from earlier computations
        frame["Home_Estimated_Turnover_Rate"] = frame.get("Home_Turnover_Rate", 0.0)
        frame["Away_Estimated_Turnover_Rate"] = frame.get("Away_Turnover_Rate", 0.0)
        frame["Home_Estimated_ORB_Rate"] = frame.get("Home_ORB_Rate", 0.0)
        frame["Away_Estimated_ORB_Rate"] = frame.get("Away_ORB_Rate", 0.0)

        return frame


__all__ = ["FeatureEngineer", "TeamStats"]
