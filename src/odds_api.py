from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import requests


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"


class OddsAPIError(RuntimeError):
    pass


class OddsAPIClient:
    def __init__(self, api_key: str | None = None, timeout: int = 25):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        self.timeout = timeout

    @property
    def has_key(self) -> bool:
        return bool(self.api_key and self.api_key != "PASTE_YOUR_KEY_HERE")

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        if not self.has_key:
            raise OddsAPIError("Missing ODDS_API_KEY. Add it to .streamlit/secrets.toml or your environment.")
        params = params.copy() if params else {}
        params["apiKey"] = self.api_key
        url = f"{BASE_URL}{path}"
        response = requests.get(url, params=params, timeout=self.timeout)
        if response.status_code >= 400:
            raise OddsAPIError(f"Odds API error {response.status_code}: {response.text[:500]}")
        return response.json()

    def get_mlb_odds(
        self,
        markets: Iterable[str] = ("h2h",),
        regions: str = "us",
        odds_format: str = "american",
        bookmakers: str | None = None,
    ) -> list[dict]:
        params = {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        data = self._get(f"/sports/{SPORT_KEY}/odds", params=params)
        if not isinstance(data, list):
            raise OddsAPIError("Expected list response from odds endpoint.")
        return data

    def get_mlb_events(self) -> list[dict]:
        data = self._get(f"/sports/{SPORT_KEY}/events")
        if not isinstance(data, list):
            raise OddsAPIError("Expected list response from events endpoint.")
        return data

    def get_event_odds(
        self,
        event_id: str,
        markets: Iterable[str],
        regions: str = "us",
        odds_format: str = "american",
        bookmakers: str | None = None,
    ) -> dict:
        params = {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        data = self._get(f"/sports/{SPORT_KEY}/events/{event_id}/odds", params=params)
        if not isinstance(data, dict):
            raise OddsAPIError("Expected object response from event odds endpoint.")
        return data


def load_sample_odds(path: str | Path = "data/sample_mlb_odds.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
