"""Pull in recent NBA data so the models can be refreshed on demand."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import sqlite3
from sbrscrape import Scoreboard

from src.Utils.tools import get_json_data, to_data_frame


STATS_URL = (
    "https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=10%2F01%2F{start_year}"
    "&DateTo={month}%2F{day}%2F{year}&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00"
    "&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame"
    "&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType=Regular%20Season"
    "&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="
)


TEAM_NAME_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
    "New Jersey Nets": "Brooklyn Nets",
    "New Orleans Hornets": "New Orleans Pelicans",
}


def _normalize_team(team_name: str) -> str:
    return TEAM_NAME_ALIASES.get(team_name, team_name)


def _fetch_team_snapshot(target_date: date, season: str) -> pd.DataFrame:
    month = f"{target_date.month:02d}"
    day = f"{target_date.day:02d}"
    year = target_date.year
    start_year = int(season.split("-")[0])
    url = STATS_URL.format(
        start_year=start_year,
        month=month,
        day=day,
        year=year,
        season=season,
    )
    raw = get_json_data(url)
    frame = to_data_frame(raw)
    return frame


def _scoreboard_games(target_date: date) -> List[dict]:
    board = Scoreboard(date=target_date)
    if not hasattr(board, "games"):
        return []
    return board.games


@dataclass
class UpToDateDataManager:
    dataset_path: Path = Path("Data/dataset.sqlite")
    table_name: str = "dataset_2012-24_new"
    season: str = "2024-25"

    def _load_existing_dataset(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            return pd.DataFrame()
        con = sqlite3.connect(self.dataset_path)
        try:
            frame = pd.read_sql_query(f'SELECT * FROM "{self.table_name}"', con)
        finally:
            con.close()
        return frame

    def _write_dataset(self, frame: pd.DataFrame) -> None:
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.dataset_path)
        try:
            frame.to_sql(self.table_name, con, if_exists="replace", index=False)
        finally:
            con.close()

    def refresh_recent_data(
        self,
        days_back: int = 3,
        sportsbook: str = "fanduel",
    ) -> pd.DataFrame:
        """Fetch the last ``days_back`` worth of games and persist them.

        Returns the rows that were appended so the caller can decide whether it
        should trigger a model refresh.
        """

        today = date.today()
        start = today - timedelta(days=days_back)
        existing = self._load_existing_dataset()

        teams_last_played: Dict[str, date] = {}
        if not existing.empty:
            for _, row in existing.sort_values("Date").iterrows():
                for team_col, date_col in (("TEAM_NAME", "Date"), ("TEAM_NAME.1", "Date.1")):
                    team = row.get(team_col)
                    played = row.get(date_col)
                    if isinstance(played, str):
                        try:
                            played_date = datetime.fromisoformat(played).date()
                        except ValueError:
                            continue
                        teams_last_played[team] = played_date

        new_rows: List[pd.Series] = []

        target_date = start
        while target_date <= today:
            games = _scoreboard_games(target_date)
            if not games:
                target_date += timedelta(days=1)
                continue

            snapshot = _fetch_team_snapshot(target_date, self.season)
            if snapshot.empty:
                target_date += timedelta(days=1)
                continue

            for game in games:
                home_team = _normalize_team(game.get("home_team"))
                away_team = _normalize_team(game.get("away_team"))

                try:
                    home_stats = snapshot.loc[snapshot["TEAM_NAME"] == home_team].iloc[0]
                    away_stats = snapshot.loc[snapshot["TEAM_NAME"] == away_team].iloc[0]
                except IndexError:
                    continue

                home_last = teams_last_played.get(home_team)
                away_last = teams_last_played.get(away_team)
                home_rest = (target_date - home_last).days if home_last else 7
                away_rest = (target_date - away_last).days if away_last else 7
                teams_last_played[home_team] = target_date
                teams_last_played[away_team] = target_date

                total_points = (game.get("home_score") or 0) + (game.get("away_score") or 0)
                margin = (game.get("home_score") or 0) - (game.get("away_score") or 0)
                ou_line = None
                if game.get("total") and sportsbook in game["total"]:
                    ou_line = game["total"][sportsbook]

                win_flag = 1 if margin > 0 else 0
                if ou_line is not None:
                    if total_points > ou_line:
                        ou_cover = 1
                    elif total_points < ou_line:
                        ou_cover = 0
                    else:
                        ou_cover = 2
                else:
                    ou_cover = 2

                combined = pd.concat([
                    home_stats,
                    away_stats.rename(lambda col: f"{col}.1"),
                ])
                combined["Date"] = str(target_date)
                combined["Date.1"] = str(target_date)
                combined["Score"] = total_points
                combined["Home-Team-Win"] = win_flag
                combined["OU"] = ou_line if ou_line is not None else 0
                combined["OU-Cover"] = ou_cover
                combined["Days-Rest-Home"] = home_rest
                combined["Days-Rest-Away"] = away_rest
                new_rows.append(combined)

            target_date += timedelta(days=1)

        if not new_rows:
            return pd.DataFrame()

        new_frame = pd.DataFrame(new_rows)
        if existing.empty:
            combined = new_frame
        else:
            new_frame = new_frame.reindex(columns=existing.columns, fill_value=0)
            combined = pd.concat([existing, new_frame], ignore_index=True)
            combined.drop_duplicates(subset=["Date", "TEAM_NAME", "TEAM_NAME.1"], keep="last", inplace=True)

        self._write_dataset(combined)
        return new_frame

