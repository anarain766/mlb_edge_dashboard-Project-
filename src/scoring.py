from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import pandas as pd


def american_to_implied(odds: float | int | None) -> Optional[float]:
    """Convert American odds to raw implied probability, including vig."""
    if odds is None or pd.isna(odds) or odds == 0:
        return None
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def american_to_profit_per_unit(odds: float | int | None) -> Optional[float]:
    """Profit returned on a $1 stake, excluding stake."""
    if odds is None or pd.isna(odds) or odds == 0:
        return None
    odds = float(odds)
    if odds > 0:
        return odds / 100
    return 100 / abs(odds)


def prob_to_american(prob: float | None) -> Optional[int]:
    """Convert probability to fair American odds."""
    if prob is None or pd.isna(prob) or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))


def expected_value_per_unit(fair_prob: float | None, odds: float | int | None) -> Optional[float]:
    """Expected profit/loss per $1 staked."""
    if fair_prob is None or odds is None or pd.isna(fair_prob) or pd.isna(odds):
        return None
    profit = american_to_profit_per_unit(odds)
    if profit is None:
        return None
    return fair_prob * profit - (1 - fair_prob)


def grade_play(edge_pct: float | None, ev_pct: float | None) -> str:
    if edge_pct is None or pd.isna(edge_pct) or ev_pct is None or pd.isna(ev_pct):
        return "No play"
    if edge_pct >= 0.06 and ev_pct >= 0.08:
        return "A"
    if edge_pct >= 0.04 and ev_pct >= 0.05:
        return "B"
    if edge_pct >= 0.025 and ev_pct >= 0.025:
        return "C"
    if edge_pct >= 0.015 and ev_pct >= 0.01:
        return "Lean"
    return "No play"


def flatten_h2h_odds(events: list[dict]) -> pd.DataFrame:
    """Flatten The Odds API h2h market response into one row per game/team/book."""
    rows: list[dict] = []
    for event in events:
        event_id = event.get("id")
        home = event.get("home_team")
        away = event.get("away_team")
        commence_time = event.get("commence_time")
        for book in event.get("bookmakers", []) or []:
            book_title = book.get("title") or book.get("key")
            book_key = book.get("key")
            for market in book.get("markets", []) or []:
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", []) or []
                for outcome in outcomes:
                    team = outcome.get("name")
                    price = outcome.get("price")
                    opponent = away if team == home else home
                    rows.append(
                        {
                            "event_id": event_id,
                            "commence_time": commence_time,
                            "home_team": home,
                            "away_team": away,
                            "team": team,
                            "opponent": opponent,
                            "is_home": team == home,
                            "book": book_title,
                            "book_key": book_key,
                            "price": price,
                            "raw_implied_prob": american_to_implied(price),
                        }
                    )
    return pd.DataFrame(rows)


def add_consensus_fair_prob(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate fair win probability from no-vig consensus across books.

    This is not a custom baseball model yet. It finds market-value by removing
    two-way vig book-by-book, then comparing the best available price to the
    broader market consensus.
    """
    if odds_df.empty:
        return odds_df

    df = odds_df.copy()
    df["game"] = df["away_team"] + " @ " + df["home_team"]

    # Calculate no-vig probabilities within each event/book when both sides exist.
    no_vig_rows = []
    for (event_id, book), grp in df.groupby(["event_id", "book"], dropna=False):
        if len(grp) < 2:
            continue
        total_raw = grp["raw_implied_prob"].sum()
        if total_raw <= 0:
            continue
        for idx, row in grp.iterrows():
            no_vig_rows.append(
                {
                    "idx": idx,
                    "book_no_vig_prob": row["raw_implied_prob"] / total_raw,
                    "book_hold_pct": total_raw - 1,
                }
            )

    if no_vig_rows:
        nv = pd.DataFrame(no_vig_rows).set_index("idx")
        df = df.join(nv)
    else:
        df["book_no_vig_prob"] = pd.NA
        df["book_hold_pct"] = pd.NA

    consensus = (
        df.groupby(["event_id", "team"], as_index=False)["book_no_vig_prob"]
        .median()
        .rename(columns={"book_no_vig_prob": "fair_prob"})
    )
    df = df.merge(consensus, on=["event_id", "team"], how="left")
    df["fair_american"] = df["fair_prob"].apply(prob_to_american)
    return df


def best_moneyline_plays(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per team/game using the best available moneyline price."""
    if odds_df.empty:
        return odds_df
    df = add_consensus_fair_prob(odds_df)

    # For American odds, higher number is always better for the bettor (+150 > +130, -110 > -130).
    idx = df.groupby(["event_id", "team"], dropna=False)["price"].idxmax()
    best = df.loc[idx].copy().reset_index(drop=True)
    best["best_price"] = best["price"]
    best["best_book"] = best["book"]
    best["best_implied_prob"] = best["best_price"].apply(american_to_implied)
    best["edge_pct"] = best["fair_prob"] - best["best_implied_prob"]
    best["ev_per_$1"] = best.apply(lambda r: expected_value_per_unit(r["fair_prob"], r["best_price"]), axis=1)
    best["ev_pct"] = best["ev_per_$1"]
    best["grade"] = best.apply(lambda r: grade_play(r["edge_pct"], r["ev_pct"]), axis=1)
    best["game"] = best["away_team"] + " @ " + best["home_team"]
    best["play"] = best["team"] + " ML " + best["best_price"].apply(lambda x: f"{int(x):+d}" if pd.notna(x) else "")
    return best.sort_values(["ev_pct", "edge_pct"], ascending=False)


def flatten_player_prop_odds(event: dict, market_key: str) -> pd.DataFrame:
    """Flatten player prop market response into player/outcome rows.

    For batter_home_runs, many books return Over/Under lines; some may return Yes/No.
    This keeps all outcomes so the app can filter for Over/Yes style entries.
    """
    rows: list[dict] = []
    event_id = event.get("id")
    home = event.get("home_team")
    away = event.get("away_team")
    commence_time = event.get("commence_time")
    for book in event.get("bookmakers", []) or []:
        book_title = book.get("title") or book.get("key")
        for market in book.get("markets", []) or []:
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes", []) or []:
                price = outcome.get("price")
                rows.append(
                    {
                        "event_id": event_id,
                        "game": f"{away} @ {home}",
                        "commence_time": commence_time,
                        "book": book_title,
                        "market": market_key,
                        "player": outcome.get("description") or outcome.get("name"),
                        "outcome": outcome.get("name"),
                        "point": outcome.get("point"),
                        "price": price,
                        "raw_implied_prob": american_to_implied(price),
                    }
                )
    return pd.DataFrame(rows)


def best_prop_prices(prop_df: pd.DataFrame) -> pd.DataFrame:
    """Rank prop rows by best available price.

    This is line-shopping, not a true HR probability model yet. Later we can merge
    batter/pitcher/park/weather features and replace consensus with a model.
    """
    if prop_df.empty:
        return prop_df
    df = prop_df.copy()
    # Keep HR over/yes outcomes. The API can return either Over/Under or Yes/No style names.
    mask = df["outcome"].str.lower().isin(["over", "yes"]) | df["outcome"].str.lower().str.contains("over|yes", na=False)
    df = df.loc[mask].copy()
    if df.empty:
        return df
    idx = df.groupby(["event_id", "player", "market", "point"], dropna=False)["price"].idxmax()
    best = df.loc[idx].copy().reset_index(drop=True)
    best["best_price"] = best["price"]
    best["best_book"] = best["book"]
    best["play"] = best["player"] + " HR " + best["best_price"].apply(lambda x: f"{int(x):+d}" if pd.notna(x) else "")
    return best.sort_values("best_price", ascending=False)


def apply_mlb_factor_adjustments(
    plays_df: pd.DataFrame,
    features_df: pd.DataFrame,
    factor_influence: float = 0.60,
) -> pd.DataFrame:
    """Blend market fair probability with MLB factor adjustments.

    The market consensus remains the anchor. The feature frame supplies a small
    probability-point nudge from record, offense, pitching, starter, and home-field
    factors. Missing features reduce the confidence multiplier automatically.
    """
    if plays_df.empty:
        return plays_df

    out = plays_df.copy()
    out["market_fair_prob"] = out["fair_prob"]

    if features_df is None or features_df.empty:
        out["model_prob_delta"] = 0.0
        out["factor_adjustment"] = 0.0
        out["factor_confidence"] = 0.0
        out["factor_summary"] = "Market only"
        return out

    keep_cols = [
        "event_id",
        "team",
        "probable_pitcher",
        "opponent_probable_pitcher",
        "venue",
        "official_status",
        "team_win_pct",
        "opp_team_win_pct",
        "team_ops",
        "opp_team_ops",
        "team_era",
        "opp_team_era",
        "sp_era",
        "opp_sp_era",
        "sp_whip",
        "opp_sp_whip",
        "record_adj",
        "offense_adj",
        "team_pitching_adj",
        "starter_adj",
        "home_adj",
        "factor_adjustment",
        "factor_confidence",
        "factor_count",
        "factor_summary",
    ]
    features = features_df[[c for c in keep_cols if c in features_df.columns]].copy()
    out = out.merge(features, on=["event_id", "team"], how="left")

    raw_adjustment = out.get("factor_adjustment", pd.Series(0, index=out.index)).fillna(0).astype(float)
    confidence = out.get("factor_confidence", pd.Series(0, index=out.index)).fillna(0).astype(float)
    influence = max(0.0, min(1.0, float(factor_influence)))
    out["model_prob_delta"] = raw_adjustment * confidence * influence
    out["fair_prob"] = (out["market_fair_prob"].astype(float) + out["model_prob_delta"]).clip(0.05, 0.95)
    out["fair_american"] = out["fair_prob"].apply(prob_to_american)
    out["edge_pct"] = out["fair_prob"] - out["best_implied_prob"]
    out["ev_per_$1"] = out.apply(lambda r: expected_value_per_unit(r["fair_prob"], r["best_price"]), axis=1)
    out["ev_pct"] = out["ev_per_$1"]
    out["grade"] = out.apply(lambda r: grade_play(r["edge_pct"], r["ev_pct"]), axis=1)
    out["factor_summary"] = out.get("factor_summary", pd.Series("Market only", index=out.index)).fillna("Market only")
    return out.sort_values(["ev_pct", "edge_pct"], ascending=False)
