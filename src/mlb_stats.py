from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import math
import re

import pandas as pd
import requests


BASE_URL = "https://statsapi.mlb.com/api/v1"


TEAM_ALIASES = {
    "athletics": "athletics",
    "oakland athletics": "athletics",
    "sacramento athletics": "athletics",
    "a's": "athletics",
    "as": "athletics",
    "az diamondbacks": "arizona diamondbacks",
    "chi cubs": "chicago cubs",
    "chi white sox": "chicago white sox",
    "cws": "chicago white sox",
    "la dodgers": "los angeles dodgers",
    "los angeles angels": "los angeles angels",
    "la angels": "los angeles angels",
    "ny mets": "new york mets",
    "ny yankees": "new york yankees",
    "sf giants": "san francisco giants",
    "tb rays": "tampa bay rays",
    "washington nationals": "washington nationals",
    "wsh nationals": "washington nationals",
}


def normalize_team_name(name: str | None) -> str:
    """Normalize sportsbook and MLB team names for safer matching."""
    if not name:
        return ""
    s = name.lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return TEAM_ALIASES.get(s, s)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text in {"-", ".---", "-.--"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _pct(value: Any) -> float | None:
    f = _to_float(value)
    if f is None:
        return None
    # MLB API can return .540 as a string or 0.540; both parse the same.
    return f


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class MLBStatsClient:
    """Small wrapper around MLB's public stats API.

    The app treats these data as an enrichment layer. If any endpoint fails, the
    dashboard falls back to the market-only model instead of breaking.
    """

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = requests.get(f"{BASE_URL}{path}", params=params or {}, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected JSON object for {path}")
        return data

    def schedule_by_date(self, date: str) -> pd.DataFrame:
        params = {
            "sportId": 1,
            "date": date,
            "hydrate": "probablePitcher,team,venue",
        }
        data = self._get("/schedule", params=params)
        rows: list[dict] = []
        for day in data.get("dates", []) or []:
            for game in day.get("games", []) or []:
                teams = game.get("teams", {}) or {}
                home = ((teams.get("home") or {}).get("team") or {}).get("name")
                away = ((teams.get("away") or {}).get("team") or {}).get("name")
                home_pp = (teams.get("home") or {}).get("probablePitcher") or {}
                away_pp = (teams.get("away") or {}).get("probablePitcher") or {}
                venue = game.get("venue") or {}
                status = game.get("status") or {}
                base = {
                    "official_game_pk": game.get("gamePk"),
                    "official_game_date": game.get("gameDate"),
                    "official_status": status.get("detailedState") or status.get("abstractGameState"),
                    "venue": venue.get("name"),
                    "home_team_official": home,
                    "away_team_official": away,
                    "home_norm": normalize_team_name(home),
                    "away_norm": normalize_team_name(away),
                }
                rows.append(
                    {
                        **base,
                        "team_norm": normalize_team_name(home),
                        "opponent_norm": normalize_team_name(away),
                        "probable_pitcher": home_pp.get("fullName"),
                        "probable_pitcher_id": home_pp.get("id"),
                        "opponent_probable_pitcher": away_pp.get("fullName"),
                        "opponent_probable_pitcher_id": away_pp.get("id"),
                        "official_is_home": True,
                    }
                )
                rows.append(
                    {
                        **base,
                        "team_norm": normalize_team_name(away),
                        "opponent_norm": normalize_team_name(home),
                        "probable_pitcher": away_pp.get("fullName"),
                        "probable_pitcher_id": away_pp.get("id"),
                        "opponent_probable_pitcher": home_pp.get("fullName"),
                        "opponent_probable_pitcher_id": home_pp.get("id"),
                        "official_is_home": False,
                    }
                )
        return pd.DataFrame(rows)

    def standings(self, season: int) -> pd.DataFrame:
        params = {
            "leagueId": "103,104",
            "season": season,
            "standingsTypes": "regularSeason",
            "hydrate": "team",
        }
        data = self._get("/standings", params=params)
        rows: list[dict] = []
        for record_group in data.get("records", []) or []:
            for team_record in record_group.get("teamRecords", []) or []:
                team = team_record.get("team") or {}
                rows.append(
                    {
                        "team_norm": normalize_team_name(team.get("name")),
                        "team_official": team.get("name"),
                        "wins": _to_float(team_record.get("wins")),
                        "losses": _to_float(team_record.get("losses")),
                        "team_win_pct": _pct(team_record.get("winningPercentage")),
                        "division_rank": team_record.get("divisionRank"),
                        "league_rank": team_record.get("leagueRank"),
                    }
                )
        return pd.DataFrame(rows).drop_duplicates("team_norm") if rows else pd.DataFrame()

    def team_stats(self, season: int, group: str) -> pd.DataFrame:
        params = {"stats": "season", "group": group, "sportIds": 1, "season": season}
        data = self._get("/teams/stats", params=params)
        rows: list[dict] = []
        for stat_block in data.get("stats", []) or []:
            for split in stat_block.get("splits", []) or []:
                team = split.get("team") or {}
                stat = split.get("stat") or {}
                row = {"team_norm": normalize_team_name(team.get("name")), "team_official": team.get("name")}
                if group == "hitting":
                    row.update(
                        {
                            "team_ops": _to_float(stat.get("ops")),
                            "team_avg": _to_float(stat.get("avg")),
                            "team_obp": _to_float(stat.get("obp")),
                            "team_slg": _to_float(stat.get("slg")),
                            "team_runs": _to_float(stat.get("runs")),
                            "team_home_runs": _to_float(stat.get("homeRuns")),
                        }
                    )
                elif group == "pitching":
                    row.update(
                        {
                            "team_era": _to_float(stat.get("era")),
                            "team_whip": _to_float(stat.get("whip")),
                            "team_pitching_home_runs": _to_float(stat.get("homeRuns")),
                            "team_strikeouts": _to_float(stat.get("strikeOuts")),
                            "team_walks": _to_float(stat.get("baseOnBalls")),
                        }
                    )
                rows.append(row)
        return pd.DataFrame(rows).drop_duplicates("team_norm") if rows else pd.DataFrame()

    def pitcher_stats(self, player_id: int | str | None, season: int) -> dict:
        if not player_id:
            return {}
        data = self._get(f"/people/{player_id}/stats", params={"stats": "season", "group": "pitching", "season": season})
        for stat_block in data.get("stats", []) or []:
            for split in stat_block.get("splits", []) or []:
                stat = split.get("stat") or {}
                return {
                    "sp_era": _to_float(stat.get("era")),
                    "sp_whip": _to_float(stat.get("whip")),
                    "sp_ip": _to_float(stat.get("inningsPitched")),
                    "sp_hr": _to_float(stat.get("homeRuns")),
                    "sp_k": _to_float(stat.get("strikeOuts")),
                    "sp_bb": _to_float(stat.get("baseOnBalls")),
                }
        return {}


def _merge_team_stats(base: pd.DataFrame, stats: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    if base.empty or stats.empty:
        return base
    stats = stats.copy()
    rename = {c: f"{prefix}{c}" for c in stats.columns if c != "team_norm"}
    stats = stats.rename(columns=rename)
    return base.merge(stats, on="team_norm", how="left")


def _attach_opponent_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    subset_cols = ["event_id", "team_norm"] + [c for c in cols if c in df.columns]
    opp = df[subset_cols].copy().rename(columns={"team_norm": "opponent_norm"})
    rename = {c: f"opp_{c}" for c in cols if c in opp.columns}
    opp = opp.rename(columns=rename)
    return df.merge(opp, on=["event_id", "opponent_norm"], how="left")


def _calc_adjustments(row: pd.Series) -> dict:
    """Create probability-point adjustments from simple baseball factors.

    These are intentionally small nudges around the market consensus. They are
    not meant to overrule the market, especially when data is missing.
    """
    components: dict[str, float | None] = {}

    team_win = row.get("team_win_pct")
    opp_win = row.get("opp_team_win_pct")
    if pd.notna(team_win) and pd.notna(opp_win):
        components["record_adj"] = _clip((float(team_win) - float(opp_win)) * 0.18, -0.035, 0.035)
    else:
        components["record_adj"] = None

    team_ops = row.get("team_ops")
    opp_ops = row.get("opp_team_ops")
    if pd.notna(team_ops) and pd.notna(opp_ops):
        # 50 OPS points is a meaningful offensive gap; convert that to about 1.5 pct points.
        components["offense_adj"] = _clip((float(team_ops) - float(opp_ops)) * 0.30, -0.025, 0.025)
    else:
        components["offense_adj"] = None

    team_era = row.get("team_era")
    opp_era = row.get("opp_team_era")
    if pd.notna(team_era) and pd.notna(opp_era):
        # Lower ERA is better, so opponent ERA minus team ERA helps this team.
        components["team_pitching_adj"] = _clip((float(opp_era) - float(team_era)) * 0.010, -0.025, 0.025)
    else:
        components["team_pitching_adj"] = None

    sp_era = row.get("sp_era")
    opp_sp_era = row.get("opp_sp_era")
    sp_whip = row.get("sp_whip")
    opp_sp_whip = row.get("opp_sp_whip")
    starter_parts: list[float] = []
    if pd.notna(sp_era) and pd.notna(opp_sp_era):
        starter_parts.append(_clip((float(opp_sp_era) - float(sp_era)) * 0.012, -0.030, 0.030))
    if pd.notna(sp_whip) and pd.notna(opp_sp_whip):
        starter_parts.append(_clip((float(opp_sp_whip) - float(sp_whip)) * 0.025, -0.020, 0.020))
    components["starter_adj"] = sum(starter_parts) if starter_parts else None

    is_home = row.get("is_home")
    if pd.notna(is_home):
        components["home_adj"] = 0.018 if bool(is_home) else -0.018
    else:
        components["home_adj"] = None

    usable = [v for v in components.values() if v is not None and pd.notna(v)]
    raw_total = float(sum(usable)) if usable else 0.0
    # Keep the total feature nudge modest because the betting market is the anchor.
    total = _clip(raw_total, -0.075, 0.075)
    confidence = len(usable) / len(components)
    return {**components, "factor_adjustment": total, "factor_confidence": confidence, "factor_count": len(usable)}


def build_mlb_feature_frame(events: list[dict], season: int | None = None) -> pd.DataFrame:
    """Build one row per odds event/team with MLB factors merged in."""
    if not events:
        return pd.DataFrame()

    if season is None:
        # Use the first event's year where possible.
        try:
            season = pd.to_datetime(events[0].get("commence_time"), utc=True).year
        except Exception:
            season = datetime.now(timezone.utc).year

    # Date of first event. If the odds call returns a multi-day board, this still
    # covers the normal daily use case; unmatched games simply keep null factors.
    try:
        date = pd.to_datetime(events[0].get("commence_time"), utc=True).date().isoformat()
    except Exception:
        date = datetime.now(timezone.utc).date().isoformat()

    rows: list[dict] = []
    for event in events:
        event_id = event.get("id")
        home = event.get("home_team")
        away = event.get("away_team")
        commence_time = event.get("commence_time")
        rows.append(
            {
                "event_id": event_id,
                "commence_time": commence_time,
                "team": home,
                "opponent": away,
                "team_norm": normalize_team_name(home),
                "opponent_norm": normalize_team_name(away),
                "is_home": True,
                "game": f"{away} @ {home}",
            }
        )
        rows.append(
            {
                "event_id": event_id,
                "commence_time": commence_time,
                "team": away,
                "opponent": home,
                "team_norm": normalize_team_name(away),
                "opponent_norm": normalize_team_name(home),
                "is_home": False,
                "game": f"{away} @ {home}",
            }
        )
    df = pd.DataFrame(rows)

    client = MLBStatsClient()

    # Schedule / probable pitchers.
    try:
        schedule = client.schedule_by_date(date)
    except Exception:
        schedule = pd.DataFrame()

    if not schedule.empty:
        sched_cols = [
            "home_norm",
            "away_norm",
            "team_norm",
            "opponent_norm",
            "official_game_pk",
            "official_status",
            "venue",
            "probable_pitcher",
            "probable_pitcher_id",
            "opponent_probable_pitcher",
            "opponent_probable_pitcher_id",
        ]
        df = df.merge(schedule[[c for c in sched_cols if c in schedule.columns]], on=["team_norm", "opponent_norm"], how="left")

    # Standings and team season stats.
    for getter, prefix in [
        (lambda: client.standings(season), ""),
        (lambda: client.team_stats(season, "hitting"), ""),
        (lambda: client.team_stats(season, "pitching"), ""),
    ]:
        try:
            stats = getter()
        except Exception:
            stats = pd.DataFrame()
        if not stats.empty:
            stats = stats.drop(columns=["team_official"], errors="ignore")
            df = _merge_team_stats(df, stats, prefix=prefix)

    opponent_cols = [
        "team_win_pct",
        "wins",
        "losses",
        "team_ops",
        "team_avg",
        "team_obp",
        "team_slg",
        "team_home_runs",
        "team_era",
        "team_whip",
        "team_pitching_home_runs",
    ]
    df = _attach_opponent_columns(df, opponent_cols)

    # Probable pitcher season stat enrichment. Keep best-effort and cached by Streamlit at caller level.
    pitcher_cache: dict[Any, dict] = {}
    for col_id, prefix in [("probable_pitcher_id", ""), ("opponent_probable_pitcher_id", "opp_")]:
        stat_rows: list[dict] = []
        for pid in df.get(col_id, pd.Series(dtype=object)).dropna().unique().tolist():
            if pid not in pitcher_cache:
                try:
                    pitcher_cache[pid] = client.pitcher_stats(pid, season)
                except Exception:
                    pitcher_cache[pid] = {}
            stat = pitcher_cache[pid].copy()
            stat[col_id] = pid
            stat_rows.append(stat)
        if stat_rows:
            pstats = pd.DataFrame(stat_rows)
            rename = {c: f"{prefix}{c}" for c in pstats.columns if c != col_id}
            pstats = pstats.rename(columns=rename)
            df = df.merge(pstats, on=col_id, how="left")

    adjustments = df.apply(_calc_adjustments, axis=1, result_type="expand")
    df = pd.concat([df, adjustments], axis=1)

    def summary(row: pd.Series) -> str:
        bits = []
        for label, col in [
            ("Record", "record_adj"),
            ("Offense", "offense_adj"),
            ("Team pitching", "team_pitching_adj"),
            ("Starter", "starter_adj"),
            ("Home", "home_adj"),
        ]:
            val = row.get(col)
            if pd.notna(val):
                bits.append(f"{label} {float(val):+.1%}")
        return "; ".join(bits) if bits else "Market only"

    df["factor_summary"] = df.apply(summary, axis=1)
    return df
