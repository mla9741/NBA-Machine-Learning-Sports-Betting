"""Client wrapper for the public Ball Don't Lie API.

The goal of the client is to provide a small surface area tailored to the
feature engineering needs of this project (injury lookups around a given
match date).  The implementation handles pagination, basic caching and
is resilient to network failures so downstream code can rely on sensible
fallback behaviour when the API is unreachable.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_LOOKBACK_DAYS = 3
SEVERITY_WEIGHTS = {
    "out": 2.0,
    "doubtful": 1.5,
    "questionable": 1.0,
    "probable": 0.5,
    "day-to-day": 0.75,
    "rest": 0.5,
}


@dataclass
class InjuryRecord:
    """Lightweight representation of an injury response payload."""

    team_name: str
    status: str
    description: str

    @property
    def weight(self) -> float:
        key = self.status.lower()
        for name, weight in SEVERITY_WEIGHTS.items():
            if name in key:
                return weight
        return 0.25  # default small penalty when information is scarce


class BallDontLieClient:
    """Minimal API client specialised for injury lookups."""

    BASE_URL = "https://www.balldontlie.io/api/v1"

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        enable_http: bool = True,
        timeout: int = 10,
        api_key: Optional[str] = None,
    ) -> None:
        self.session = session or requests.Session()
        self.enable_http = enable_http
        self.timeout = timeout
        self.api_key = api_key or os.getenv("BALLDONTLIE_API_KEY")
        if self.api_key:
            # The public Ball Don't Lie API expects the key via the
            # ``Authorization`` header.  We set it once on the session so all
            # subsequent requests inherit it automatically.
            self.session.headers.setdefault("Authorization", f"Bearer {self.api_key}")
        self._cache: Dict[Tuple[str, str], List[Dict]] = {}

    # -- HTTP helpers -------------------------------------------------
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        if not self.enable_http:
            return {}
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}

    def _iter_pages(self, endpoint: str, params: Optional[Dict] = None) -> Iterable[Dict]:
        params = params.copy() if params else {}
        params.setdefault("per_page", 100)
        page = 1
        while True:
            params["page"] = page
            payload = self._request(endpoint, params=params)
            if not payload:
                break
            data = payload.get("data", [])
            for item in data:
                yield item
            meta = payload.get("meta", {})
            total_pages = meta.get("total_pages", 1)
            if page >= total_pages:
                break
            page += 1

    # -- Public API ---------------------------------------------------
    def get_injuries(self, start_date: dt.date, end_date: dt.date) -> List[Dict]:
        """Return injury reports between the given dates (inclusive)."""

        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        cache_key = (start_str, end_str)
        if cache_key in self._cache:
            return self._cache[cache_key]

        results = list(
            self._iter_pages(
                "injuries",
                params={"start_date": start_str, "end_date": end_str},
            )
        )
        self._cache[cache_key] = results
        return results

    def _normalise_team_name(self, payload: Dict) -> str:
        team = payload.get("team", {})
        return team.get("full_name") or team.get("name") or ""

    def get_team_injury_score(
        self,
        team_name: str,
        game_date: dt.date,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> float:
        """Compute an aggregated injury severity score for ``team_name``."""

        if not team_name:
            return 0.0

        start_date = game_date - dt.timedelta(days=lookback_days)
        injuries = self.get_injuries(start_date, game_date)
        if not injuries:
            return 0.0

        score = 0.0
        for item in injuries:
            if self._normalise_team_name(item) != team_name:
                continue
            status = item.get("status", "")
            description = item.get("description", "")
            record = InjuryRecord(team_name=team_name, status=status, description=description)
            score += record.weight
        return score


__all__ = ["BallDontLieClient", "InjuryRecord", "DEFAULT_LOOKBACK_DAYS"]
