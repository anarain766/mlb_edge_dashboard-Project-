from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os

import pandas as pd
import streamlit as st

from src.odds_api import OddsAPIClient, OddsAPIError, load_sample_odds
from src.scoring import (
    flatten_h2h_odds,
    best_moneyline_plays,
    flatten_player_prop_odds,
    best_prop_prices,
)
from src.parlay import build_parlays
from src.storage import init_db, add_bet, load_bets, update_result


st.set_page_config(page_title="MLB Edge Dashboard", page_icon="⚾", layout="wide")
st.title("⚾ MLB Edge Dashboard")
st.caption("Daily moneyline value, HR prop line-shopping, parlay ideas, and bet tracking.")


@st.cache_data(ttl=300, show_spinner=False)
def get_odds_data(api_key: str | None, regions: str, bookmakers: str | None):
    client = OddsAPIClient(api_key=api_key)
    if client.has_key:
        return client.get_mlb_odds(markets=("h2h",), regions=regions, bookmakers=bookmakers or None), False
    return load_sample_odds(), True


@st.cache_data(ttl=300, show_spinner=False)
def get_event_prop_data(api_key: str | None, event_id: str, market: str, regions: str, bookmakers: str | None):
    client = OddsAPIClient(api_key=api_key)
    if not client.has_key:
        return None
    return client.get_event_odds(event_id=event_id, markets=(market,), regions=regions, bookmakers=bookmakers or None)


def get_api_key() -> str | None:
    # Streamlit secrets first, then environment variable.
    try:
        key = st.secrets.get("ODDS_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("ODDS_API_KEY")


with st.sidebar:
    st.header("Settings")
    api_key = get_api_key()
    regions = st.selectbox("Region", ["us", "us2", "uk", "eu", "au"], index=0)
    bookmakers = st.text_input(
        "Optional bookmaker keys",
        value="",
        help="Comma-separated keys like fanduel,draftkings,betmgm. Leave blank for all available books in the selected region.",
    )
    min_grade = st.selectbox("Minimum grade", ["All", "Lean", "C", "B", "A"], index=0)
    st.divider()
    st.caption("No API key? The app uses sample data so you can test the layout.")
    if st.button("Refresh odds"):
        st.cache_data.clear()
        st.rerun()

try:
    events, using_sample = get_odds_data(api_key, regions, bookmakers.strip())
except OddsAPIError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Unexpected data error: {e}")
    st.stop()

if using_sample:
    st.warning("Using sample odds. Add ODDS_API_KEY to .streamlit/secrets.toml to pull live odds.")

raw_df = flatten_h2h_odds(events)
plays_df = best_moneyline_plays(raw_df)

if not plays_df.empty:
    grade_order = {"No play": 0, "Lean": 1, "C": 2, "B": 3, "A": 4}
    plays_df["grade_rank"] = plays_df["grade"].map(grade_order).fillna(0)
    if min_grade != "All":
        plays_df = plays_df[plays_df["grade_rank"] >= grade_order[min_grade]]

ml_tab, hr_tab, parlay_tab, tracker_tab, raw_tab = st.tabs(
    ["Moneyline Board", "HR Props", "Parlay Builder", "Bet Tracker", "Raw Odds"]
)

with ml_tab:
    st.subheader("Moneyline board")
    st.write(
        "This first MVP estimates market value by removing vig from the broader market consensus, "
        "then comparing each team to the best available sportsbook price."
    )
    if plays_df.empty:
        st.info("No moneyline plays found with the current filters.")
    else:
        display = plays_df[[
            "commence_time", "game", "play", "best_book", "fair_prob", "fair_american",
            "best_implied_prob", "edge_pct", "ev_per_$1", "grade"
        ]].copy()
        display["commence_time"] = pd.to_datetime(display["commence_time"]).dt.tz_convert("America/New_York")
        st.dataframe(
            display,
            use_container_width=True,
            column_config={
                "commence_time": st.column_config.DatetimeColumn("Start ET"),
                "fair_prob": st.column_config.NumberColumn("Fair win %", format="%.1%"),
                "best_implied_prob": st.column_config.NumberColumn("Best implied %", format="%.1%"),
                "edge_pct": st.column_config.NumberColumn("Edge", format="%.2%"),
                "ev_per_$1": st.column_config.NumberColumn("EV / $1", format="$%.3f"),
                "fair_american": st.column_config.NumberColumn("Fair line"),
            },
            hide_index=True,
        )

        st.markdown("### Add a pick to tracker")
        pick_label = st.selectbox("Pick", plays_df["play"].tolist())
        selected = plays_df.loc[plays_df["play"] == pick_label].iloc[0]
        stake = st.number_input("Stake", min_value=0.0, value=10.0, step=1.0)
        notes = st.text_input("Notes", value=f"Grade {selected['grade']}; edge {selected['edge_pct']:.2%}")
        if st.button("Add ML pick"):
            add_bet(
                event_date=str(pd.to_datetime(selected["commence_time"]).date()),
                bet_type="Moneyline",
                play=selected["play"],
                book=selected["best_book"],
                odds=int(selected["best_price"]),
                stake=float(stake),
                notes=notes,
            )
            st.success("Pick added to tracker.")

with hr_tab:
    st.subheader("HR props")
    st.write(
        "This tab pulls `batter_home_runs` for one selected game and ranks the best available prices. "
        "The current version is line-shopping; the next version will add batter/pitcher/park/weather features."
    )
    if using_sample:
        st.info("HR props need a live Odds API key because player props are event-specific.")
    else:
        event_options = {
            f"{e.get('away_team')} @ {e.get('home_team')} — {e.get('commence_time')}": e.get("id")
            for e in events
        }
        if not event_options:
            st.info("No events returned.")
        else:
            selected_event_label = st.selectbox("Game", list(event_options.keys()))
            event_id = event_options[selected_event_label]
            market = st.selectbox("Market", ["batter_home_runs", "batter_home_runs_alternate", "pitcher_strikeouts"])
            if st.button("Load player props"):
                try:
                    event_props = get_event_prop_data(api_key, event_id, market, regions, bookmakers.strip())
                    prop_df = flatten_player_prop_odds(event_props, market)
                    best_props = best_prop_prices(prop_df)
                    if best_props.empty:
                        st.info("No Over/Yes prop rows found for this game/market.")
                    else:
                        st.dataframe(
                            best_props[["game", "player", "outcome", "point", "best_price", "best_book", "raw_implied_prob", "play"]],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "raw_implied_prob": st.column_config.NumberColumn("Raw implied %", format="%.1%")
                            },
                        )
                except Exception as e:
                    st.error(f"Could not load props: {e}")

with parlay_tab:
    st.subheader("Parlay builder")
    st.write("Parlays are high variance. This builder avoids duplicate games and uses the top ranked ML edges.")
    legs = st.selectbox("Legs", [2, 3], index=0)
    candidate_grades = st.multiselect("Use grades", ["A", "B", "C", "Lean"], default=["A", "B", "C", "Lean"])
    parlay_candidates = plays_df[plays_df["grade"].isin(candidate_grades)] if not plays_df.empty else plays_df
    parlays = build_parlays(parlay_candidates, legs=legs)
    if parlays.empty:
        st.info("No parlays available with the selected filters.")
    else:
        st.dataframe(
            parlays.head(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                "estimated_hit_prob": st.column_config.NumberColumn("Estimated hit %", format="%.1%"),
                "ev_per_$1": st.column_config.NumberColumn("EV / $1", format="$%.3f"),
                "parlay_price": st.column_config.NumberColumn("Parlay odds"),
            },
        )

with tracker_tab:
    st.subheader("Bet tracker")
    init_db()
    bets = load_bets()
    if bets.empty:
        st.info("No bets tracked yet.")
    else:
        settled = bets[bets["result"].isin(["win", "loss", "push"])]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tracked bets", len(bets))
        c2.metric("Settled bets", len(settled))
        c3.metric("P/L", f"${bets['profit_loss'].sum():,.2f}")
        total_staked = bets["stake"].sum()
        roi = bets["profit_loss"].sum() / total_staked if total_staked else 0
        c4.metric("ROI", f"{roi:.1%}")
        st.dataframe(bets, use_container_width=True, hide_index=True)

        st.markdown("### Update result")
        bet_id = st.number_input("Bet ID", min_value=1, step=1)
        result = st.selectbox("Result", ["pending", "win", "loss", "push"])
        profit_loss = st.number_input("Profit/Loss", value=0.0, step=1.0)
        if st.button("Update bet result"):
            update_result(int(bet_id), result, float(profit_loss))
            st.success("Bet updated.")
            st.rerun()

with raw_tab:
    st.subheader("Raw odds")
    st.dataframe(raw_df, use_container_width=True, hide_index=True)
    with st.expander("Raw JSON"):
        st.json(events)
