"""Fetch game schedules/results from the Ball Don't Lie API into SQLite."""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Tuple

import toml

# Make project root importable when running as ``python -m``
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.DataProviders.BallDontLieClient import BallDontLieClient  # noqa: E402

CONFIG_PATH = ROOT / "config.toml"
DB_PATH = ROOT / "Data" / "BallDontLieGames.sqlite"
TABLE_NAME = "games"


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            season INTEGER,
            status TEXT,
            period INTEGER,
            postseason INTEGER,
            time TEXT,
            home_team_id INTEGER,
            home_team TEXT,
            home_team_score INTEGER,
            visitor_team_id INTEGER,
            visitor_team TEXT,
            visitor_team_score INTEGER,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _iso_date(timestamp: str) -> str:
    if not timestamp:
        return ""
    if timestamp.endswith("Z"):
        timestamp = timestamp.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(timestamp).isoformat()


def _flatten_game(payload: Dict) -> Tuple:
    home = payload.get("home_team", {})
    visitor = payload.get("visitor_team", {})
    return (
        int(payload.get("id")),
        _iso_date(payload.get("date", "")),
        payload.get("season"),
        payload.get("status"),
        payload.get("period"),
        1 if payload.get("postseason") else 0,
        payload.get("time"),
        home.get("id"),
        home.get("full_name") or home.get("name"),
        payload.get("home_team_score"),
        visitor.get("id"),
        visitor.get("full_name") or visitor.get("name"),
        payload.get("visitor_team_score"),
    )


def import_games(client: BallDontLieClient, seasons: Dict[str, Dict]) -> None:
    if not seasons:
        print("No seasons configured under [ball-dont-lie-games]. Nothing to import.")
        return

    with sqlite3.connect(DB_PATH) as connection:
        _ensure_schema(connection)
        cursor = connection.cursor()

        for label, config in seasons.items():
            start = _parse_date(config["start_date"])
            end = _parse_date(config["end_date"])
            print(f"Importing Ball Don't Lie games for {label}: {start} -> {end}")

            rows = 0
            for payload in client.iter_games(start_date=start, end_date=end):
                cursor.execute(
                    f"""
                    INSERT INTO {TABLE_NAME} (
                        id, date, season, status, period, postseason, time,
                        home_team_id, home_team, home_team_score,
                        visitor_team_id, visitor_team, visitor_team_score,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        date=excluded.date,
                        season=excluded.season,
                        status=excluded.status,
                        period=excluded.period,
                        postseason=excluded.postseason,
                        time=excluded.time,
                        home_team_id=excluded.home_team_id,
                        home_team=excluded.home_team,
                        home_team_score=excluded.home_team_score,
                        visitor_team_id=excluded.visitor_team_id,
                        visitor_team=excluded.visitor_team,
                        visitor_team_score=excluded.visitor_team_score,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    _flatten_game(payload),
                )
                rows += 1

            connection.commit()
            print(f"  Imported {rows} games from Ball Don't Lie for {label}.")


def main() -> None:
    config = toml.load(CONFIG_PATH)
    seasons: Dict[str, Dict] = config.get("ball-dont-lie-games", {})
    client = BallDontLieClient()

    if not client.enable_http:
        print(
            "Warning: BALLDONTLIE_API_KEY is not configured; skipping HTTP calls.",
        )
        return

    import_games(client, seasons)


if __name__ == "__main__":
    main()
