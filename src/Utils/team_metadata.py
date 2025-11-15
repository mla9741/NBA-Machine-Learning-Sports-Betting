"""Team metadata for NBA franchises used for feature engineering.

The module exposes a `TEAM_METADATA` dictionary that maps a team name to
location and contextual information that can be used to derive travel,
altitude, and timezone based features.  A helper `get_team_metadata`
function normalises common aliases so the rest of the codebase can always
retrieve a valid metadata payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class TeamMetadata:
    """Container for team specific contextual information."""

    name: str
    arena: str
    city: str
    state: str
    latitude: float
    longitude: float
    altitude_ft: float
    timezone: str


TEAM_METADATA: Dict[str, TeamMetadata] = {
    "Atlanta Hawks": TeamMetadata(
        name="Atlanta Hawks",
        arena="State Farm Arena",
        city="Atlanta",
        state="GA",
        latitude=33.7573,
        longitude=-84.3963,
        altitude_ft=1050.0,
        timezone="America/New_York",
    ),
    "Boston Celtics": TeamMetadata(
        name="Boston Celtics",
        arena="TD Garden",
        city="Boston",
        state="MA",
        latitude=42.3663,
        longitude=-71.0622,
        altitude_ft=20.0,
        timezone="America/New_York",
    ),
    "Brooklyn Nets": TeamMetadata(
        name="Brooklyn Nets",
        arena="Barclays Center",
        city="Brooklyn",
        state="NY",
        latitude=40.6826,
        longitude=-73.9747,
        altitude_ft=50.0,
        timezone="America/New_York",
    ),
    "Charlotte Hornets": TeamMetadata(
        name="Charlotte Hornets",
        arena="Spectrum Center",
        city="Charlotte",
        state="NC",
        latitude=35.2251,
        longitude=-80.8392,
        altitude_ft=748.0,
        timezone="America/New_York",
    ),
    "Chicago Bulls": TeamMetadata(
        name="Chicago Bulls",
        arena="United Center",
        city="Chicago",
        state="IL",
        latitude=41.8807,
        longitude=-87.6742,
        altitude_ft=594.0,
        timezone="America/Chicago",
    ),
    "Cleveland Cavaliers": TeamMetadata(
        name="Cleveland Cavaliers",
        arena="Rocket Mortgage FieldHouse",
        city="Cleveland",
        state="OH",
        latitude=41.4965,
        longitude=-81.6882,
        altitude_ft=653.0,
        timezone="America/New_York",
    ),
    "Dallas Mavericks": TeamMetadata(
        name="Dallas Mavericks",
        arena="American Airlines Center",
        city="Dallas",
        state="TX",
        latitude=32.7905,
        longitude=-96.8104,
        altitude_ft=430.0,
        timezone="America/Chicago",
    ),
    "Denver Nuggets": TeamMetadata(
        name="Denver Nuggets",
        arena="Ball Arena",
        city="Denver",
        state="CO",
        latitude=39.7487,
        longitude=-105.0077,
        altitude_ft=5280.0,
        timezone="America/Denver",
    ),
    "Detroit Pistons": TeamMetadata(
        name="Detroit Pistons",
        arena="Little Caesars Arena",
        city="Detroit",
        state="MI",
        latitude=42.3410,
        longitude=-83.0551,
        altitude_ft=605.0,
        timezone="America/Detroit",
    ),
    "Golden State Warriors": TeamMetadata(
        name="Golden State Warriors",
        arena="Chase Center",
        city="San Francisco",
        state="CA",
        latitude=37.7680,
        longitude=-122.3877,
        altitude_ft=10.0,
        timezone="America/Los_Angeles",
    ),
    "Houston Rockets": TeamMetadata(
        name="Houston Rockets",
        arena="Toyota Center",
        city="Houston",
        state="TX",
        latitude=29.7508,
        longitude=-95.3621,
        altitude_ft=43.0,
        timezone="America/Chicago",
    ),
    "Indiana Pacers": TeamMetadata(
        name="Indiana Pacers",
        arena="Gainbridge Fieldhouse",
        city="Indianapolis",
        state="IN",
        latitude=39.7639,
        longitude=-86.1555,
        altitude_ft=717.0,
        timezone="America/Indiana/Indianapolis",
    ),
    "Los Angeles Clippers": TeamMetadata(
        name="Los Angeles Clippers",
        arena="Crypto.com Arena",
        city="Los Angeles",
        state="CA",
        latitude=34.0430,
        longitude=-118.2673,
        altitude_ft=305.0,
        timezone="America/Los_Angeles",
    ),
    "Los Angeles Lakers": TeamMetadata(
        name="Los Angeles Lakers",
        arena="Crypto.com Arena",
        city="Los Angeles",
        state="CA",
        latitude=34.0430,
        longitude=-118.2673,
        altitude_ft=305.0,
        timezone="America/Los_Angeles",
    ),
    "Memphis Grizzlies": TeamMetadata(
        name="Memphis Grizzlies",
        arena="FedExForum",
        city="Memphis",
        state="TN",
        latitude=35.1382,
        longitude=-90.0506,
        altitude_ft=338.0,
        timezone="America/Chicago",
    ),
    "Miami Heat": TeamMetadata(
        name="Miami Heat",
        arena="Kaseya Center",
        city="Miami",
        state="FL",
        latitude=25.7814,
        longitude=-80.1870,
        altitude_ft=7.0,
        timezone="America/New_York",
    ),
    "Milwaukee Bucks": TeamMetadata(
        name="Milwaukee Bucks",
        arena="Fiserv Forum",
        city="Milwaukee",
        state="WI",
        latitude=43.0451,
        longitude=-87.9173,
        altitude_ft=617.0,
        timezone="America/Chicago",
    ),
    "Minnesota Timberwolves": TeamMetadata(
        name="Minnesota Timberwolves",
        arena="Target Center",
        city="Minneapolis",
        state="MN",
        latitude=44.9795,
        longitude=-93.2762,
        altitude_ft=830.0,
        timezone="America/Chicago",
    ),
    "New Orleans Pelicans": TeamMetadata(
        name="New Orleans Pelicans",
        arena="Smoothie King Center",
        city="New Orleans",
        state="LA",
        latitude=29.9489,
        longitude=-90.0810,
        altitude_ft=3.0,
        timezone="America/Chicago",
    ),
    "New York Knicks": TeamMetadata(
        name="New York Knicks",
        arena="Madison Square Garden",
        city="New York",
        state="NY",
        latitude=40.7505,
        longitude=-73.9934,
        altitude_ft=66.0,
        timezone="America/New_York",
    ),
    "Oklahoma City Thunder": TeamMetadata(
        name="Oklahoma City Thunder",
        arena="Paycom Center",
        city="Oklahoma City",
        state="OK",
        latitude=35.4634,
        longitude=-97.5151,
        altitude_ft=1200.0,
        timezone="America/Chicago",
    ),
    "Orlando Magic": TeamMetadata(
        name="Orlando Magic",
        arena="Kia Center",
        city="Orlando",
        state="FL",
        latitude=28.5392,
        longitude=-81.3839,
        altitude_ft=82.0,
        timezone="America/New_York",
    ),
    "Philadelphia 76ers": TeamMetadata(
        name="Philadelphia 76ers",
        arena="Wells Fargo Center",
        city="Philadelphia",
        state="PA",
        latitude=39.9012,
        longitude=-75.1720,
        altitude_ft=36.0,
        timezone="America/New_York",
    ),
    "Phoenix Suns": TeamMetadata(
        name="Phoenix Suns",
        arena="Footprint Center",
        city="Phoenix",
        state="AZ",
        latitude=33.4457,
        longitude=-112.0712,
        altitude_ft=1086.0,
        timezone="America/Phoenix",
    ),
    "Portland Trail Blazers": TeamMetadata(
        name="Portland Trail Blazers",
        arena="Moda Center",
        city="Portland",
        state="OR",
        latitude=45.5316,
        longitude=-122.6668,
        altitude_ft=43.0,
        timezone="America/Los_Angeles",
    ),
    "Sacramento Kings": TeamMetadata(
        name="Sacramento Kings",
        arena="Golden 1 Center",
        city="Sacramento",
        state="CA",
        latitude=38.5803,
        longitude=-121.4997,
        altitude_ft=30.0,
        timezone="America/Los_Angeles",
    ),
    "San Antonio Spurs": TeamMetadata(
        name="San Antonio Spurs",
        arena="Frost Bank Center",
        city="San Antonio",
        state="TX",
        latitude=29.4269,
        longitude=-98.4375,
        altitude_ft=650.0,
        timezone="America/Chicago",
    ),
    "Toronto Raptors": TeamMetadata(
        name="Toronto Raptors",
        arena="Scotiabank Arena",
        city="Toronto",
        state="ON",
        latitude=43.6435,
        longitude=-79.3791,
        altitude_ft=251.0,
        timezone="America/Toronto",
    ),
    "Utah Jazz": TeamMetadata(
        name="Utah Jazz",
        arena="Delta Center",
        city="Salt Lake City",
        state="UT",
        latitude=40.7683,
        longitude=-111.9011,
        altitude_ft=4226.0,
        timezone="America/Denver",
    ),
    "Washington Wizards": TeamMetadata(
        name="Washington Wizards",
        arena="Capital One Arena",
        city="Washington",
        state="DC",
        latitude=38.8981,
        longitude=-77.0209,
        altitude_ft=130.0,
        timezone="America/New_York",
    ),
    "New Jersey Nets": TeamMetadata(
        name="New Jersey Nets",
        arena="Prudential Center",
        city="Newark",
        state="NJ",
        latitude=40.7336,
        longitude=-74.1710,
        altitude_ft=30.0,
        timezone="America/New_York",
    ),
    "Seattle SuperSonics": TeamMetadata(
        name="Seattle SuperSonics",
        arena="KeyArena",
        city="Seattle",
        state="WA",
        latitude=47.6205,
        longitude=-122.3510,
        altitude_ft=150.0,
        timezone="America/Los_Angeles",
    ),
    "Charlotte Bobcats": TeamMetadata(
        name="Charlotte Bobcats",
        arena="Spectrum Center",
        city="Charlotte",
        state="NC",
        latitude=35.2251,
        longitude=-80.8392,
        altitude_ft=748.0,
        timezone="America/New_York",
    ),
}


TEAM_ALIASES: Dict[str, str] = {
    "LA Clippers": "Los Angeles Clippers",
    "New Orleans Hornets": "New Orleans Pelicans",
    "Charlotte Bobcats": "Charlotte Bobcats",
    "Charlotte Hornets": "Charlotte Hornets",
    "Brooklyn Nets": "Brooklyn Nets",
    "New Jersey Nets": "New Jersey Nets",
    "Seattle SuperSonics": "Seattle SuperSonics",
}


def get_team_metadata(team_name: str) -> Optional[TeamMetadata]:
    """Return metadata for a team handling common aliases.

    Args:
        team_name: Team name as present in the dataset.

    Returns:
        A :class:`TeamMetadata` instance or ``None`` if the team is
        unknown.  The helper falls back to alias lookups before giving up
        which keeps downstream feature engineering code clean and robust.
    """

    if not team_name:
        return None

    canonical = TEAM_ALIASES.get(team_name, team_name)
    return TEAM_METADATA.get(canonical)
