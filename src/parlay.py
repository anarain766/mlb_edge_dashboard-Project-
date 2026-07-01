from __future__ import annotations

import itertools
from typing import Iterable
import pandas as pd

from .scoring import american_to_profit_per_unit, prob_to_american


def american_to_decimal(odds: float | int) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds >= 2:
        return int(round((decimal_odds - 1) * 100))
    return int(round(-100 / (decimal_odds - 1)))


def build_parlays(
    picks_df: pd.DataFrame,
    legs: int = 2,
    max_rows: int = 15,
    max_combos: int = 2500,
) -> pd.DataFrame:
    """Build simple parlays from ranked picks, avoiding duplicate games.

    max_combos prevents 6-8 leg requests from creating a large combination
    explosion on Streamlit Cloud.
    """
    if picks_df.empty or len(picks_df) < legs:
        return pd.DataFrame()

    base = picks_df.head(max_rows).copy()
    rows = []
    evaluated = 0
    for combo in itertools.combinations(base.to_dict("records"), legs):
        evaluated += 1
        if evaluated > max_combos:
            break
        event_ids = [c.get("event_id") for c in combo]
        if len(set(event_ids)) != len(event_ids):
            continue
        dec = 1.0
        fair_prob = 1.0
        ev = 0.0
        plays = []
        books = []
        for c in combo:
            dec *= american_to_decimal(c["best_price"])
            fair_prob *= float(c.get("fair_prob", 0) or 0)
            plays.append(c.get("play") or c.get("team"))
            books.append(c.get("best_book", ""))
        parlay_price = decimal_to_american(dec)
        # EV per $1 = P(win) * profit - P(lose)
        ev = fair_prob * (dec - 1) - (1 - fair_prob)
        rows.append(
            {
                "legs": legs,
                "plays": " + ".join(plays),
                "books": " / ".join(books),
                "parlay_price": parlay_price,
                "estimated_hit_prob": fair_prob,
                "ev_per_$1": ev,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ev_per_$1", ascending=False)
