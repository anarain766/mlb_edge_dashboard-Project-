from __future__ import annotations

from datetime import datetime, timezone
import os

import pandas as pd
import streamlit as st

from src.odds_api import OddsAPIClient, OddsAPIError, load_sample_odds
from src.scoring import (
    flatten_h2h_odds,
    best_moneyline_plays,
    flatten_player_prop_odds,
    best_prop_prices,
    american_to_implied,
    prob_to_american,
    apply_mlb_factor_adjustments,
)
from src.mlb_stats import build_mlb_feature_frame
from src.parlay import build_parlays
from src.storage import init_db, add_bet, load_bets, update_result


st.set_page_config(page_title="MLB Edge Dashboard", page_icon="⚾", layout="wide")
st.title("⚾ MLB Edge Dashboard")
st.caption("Daily moneyline value, MLB-factor scoring, HR prop line-shopping, parlay ideas, and bet tracking.")


ET = "America/New_York"
GRADE_ORDER = {"No play": 0, "Lean": 1, "C": 2, "B": 3, "A": 4}


def format_american(odds: float | int | None) -> str:
    if odds is None or pd.isna(odds):
        return ""
    return f"{int(round(float(odds))):+d}"


def playable_to_price(fair_prob: float | None, min_edge: float = 0.015) -> int | None:
    """Worst acceptable American price while preserving at least min_edge probability edge."""
    if fair_prob is None or pd.isna(fair_prob):
        return None
    max_implied = float(fair_prob) - min_edge
    if max_implied <= 0.01 or max_implied >= 0.99:
        return None
    return prob_to_american(max_implied)


def make_recommendation(row: pd.Series) -> str:
    grade = row.get("grade", "No play")
    edge = row.get("edge_pct", 0)
    if grade in {"A", "B"}:
        return "Play"
    if grade == "C":
        return "Small play"
    if grade == "Lean":
        return "Lean only"
    if pd.notna(edge) and edge > 0:
        return "Watch price"
    return "Pass"


def add_card_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["grade_rank"] = out["grade"].map(GRADE_ORDER).fillna(0)
    out["playable_to"] = out["fair_prob"].apply(playable_to_price)
    out["playable_to_label"] = out["playable_to"].apply(format_american)
    out["recommendation"] = out.apply(make_recommendation, axis=1)
    out["card_score"] = (
        out["grade_rank"].astype(float) * 100
        + out["edge_pct"].fillna(0).astype(float) * 1000
        + out["ev_per_$1"].fillna(0).astype(float) * 100
    )
    return out.sort_values(["grade_rank", "ev_per_$1", "edge_pct"], ascending=False)


def to_et(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(ET)


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


@st.cache_data(ttl=1800, show_spinner=False)
def get_mlb_factor_data(events: list[dict], season: int):
    return build_mlb_feature_frame(events, season=season)


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
    st.subheader("Model")
    model_mode = st.selectbox("Moneyline model", ["Market + MLB factors", "Market only"], index=0)
    model_season = st.number_input("MLB stats season", min_value=2020, max_value=2035, value=datetime.now().year, step=1)
    factor_influence = st.slider(
        "MLB factor influence",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="How much record/offense/pitching/starter/home-field factors can nudge the no-vig market probability.",
    )

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
    st.warning("Using sample odds. Add ODDS_API_KEY to Streamlit Secrets to pull live odds.")
else:
    st.success("Live odds loaded.")

raw_df = flatten_h2h_odds(events)
base_plays_df = best_moneyline_plays(raw_df)
features_df = pd.DataFrame()
model_note = "Market-only: no-vig consensus compared to best available price."

if model_mode == "Market + MLB factors" and not base_plays_df.empty:
    try:
        features_df = get_mlb_factor_data(events, int(model_season))
        if not features_df.empty:
            base_plays_df = apply_mlb_factor_adjustments(base_plays_df, features_df, factor_influence=float(factor_influence))
            model_note = (
                "Blended model: no-vig market probability nudged by MLB factors "
                "(record, offense, team pitching, probable starters, and home field)."
            )
            st.info("MLB factor model loaded. Check the Model Factors tab to see what changed each play.")
        else:
            st.warning("MLB factor data was not available, so the app is using market-only scoring.")
    except Exception as e:
        st.warning(f"MLB factor data could not be loaded, so the app is using market-only scoring. Details: {e}")

plays_df = add_card_columns(base_plays_df)

if not plays_df.empty and min_grade != "All":
    plays_df = plays_df[plays_df["grade_rank"] >= GRADE_ORDER[min_grade]]

today_tab, ml_tab, factors_tab, hr_tab, parlay_tab, tracker_tab, raw_tab = st.tabs(
    ["Today's Card", "Moneyline Board", "Model Factors", "HR Props", "Parlay Builder", "Bet Tracker", "Raw Odds"]
)

with today_tab:
    st.subheader("Today's card")
    st.write(
        "This tab turns the odds board into a short betting card. "
        f"Current scoring: {model_note} "
        "Only play a pick if the price is still at or better than the playable-to number."
    )

    if plays_df.empty:
        st.info("No moneyline plays found with the current filters.")
    else:
        card_df = plays_df.copy()
        card_df["commence_time_et"] = to_et(card_df["commence_time"])
        playable_df = card_df[card_df["grade"].isin(["A", "B", "C", "Lean"])].copy()
        top_df = playable_df.head(8)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Playable MLs", len(playable_df))
        c2.metric("A/B plays", int(card_df["grade"].isin(["A", "B"]).sum()))
        best_edge = card_df["edge_pct"].max() if not card_df.empty else 0
        c3.metric("Best edge", f"{best_edge:.2%}")
        next_start = card_df["commence_time_et"].min()
        c4.metric("Next game ET", next_start.strftime("%-I:%M %p") if pd.notna(next_start) else "")

        st.markdown("### Best ML plays")
        if top_df.empty:
            st.info("No graded ML plays right now. Check again closer to lineups or after market movement.")
        else:
            show_cols = [
                "commence_time_et",
                "game",
                "play",
                "best_book",
                "fair_prob",
                "market_fair_prob",
                "model_prob_delta",
                "best_implied_prob",
                "edge_pct",
                "ev_per_$1",
                "playable_to_label",
                "grade",
                "recommendation",
                "factor_summary",
            ]
            show = top_df[[c for c in show_cols if c in top_df.columns]].copy()
            st.dataframe(
                show,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "commence_time_et": st.column_config.DatetimeColumn("Start ET"),
                    "best_book": "Best book",
                    "fair_prob": st.column_config.NumberColumn("Model win %", format="%.1%"),
                    "market_fair_prob": st.column_config.NumberColumn("Market win %", format="%.1%"),
                    "model_prob_delta": st.column_config.NumberColumn("MLB factor +/-", format="%.2%"),
                    "best_implied_prob": st.column_config.NumberColumn("Best implied %", format="%.1%"),
                    "edge_pct": st.column_config.NumberColumn("Edge", format="%.2%"),
                    "ev_per_$1": st.column_config.NumberColumn("EV / $1", format="$%.3f"),
                    "playable_to_label": "Playable to",
                },
            )

            st.caption(
                "Grade guide: A/B = strongest edges, C = smaller single, Lean = watch/small only. "
                "Playable-to keeps roughly a 1.5 percentage-point edge versus the current model fair probability."
            )

            st.markdown("### Quick add to tracker")
            pick_label = st.selectbox("Card pick", top_df["play"].tolist(), key="card_pick")
            selected = top_df.loc[top_df["play"] == pick_label].iloc[0]
            stake = st.number_input("Stake", min_value=0.0, value=10.0, step=1.0, key="card_stake")
            notes = st.text_input(
                "Notes",
                value=(
                    f"{selected['recommendation']}; grade {selected['grade']}; "
                    f"edge {selected['edge_pct']:.2%}; playable to {selected['playable_to_label']}"
                ),
                key="card_notes",
            )
            if st.button("Add card pick"):
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

        st.markdown("### Small parlay ideas")
        parlay_pool = playable_df[playable_df["grade"].isin(["A", "B", "C", "Lean"])]
        two_leg = build_parlays(parlay_pool, legs=2).head(5)
        if two_leg.empty:
            st.info("No 2-leg parlay ideas available from the current graded plays.")
        else:
            st.dataframe(
                two_leg,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "estimated_hit_prob": st.column_config.NumberColumn("Estimated hit %", format="%.1%"),
                    "ev_per_$1": st.column_config.NumberColumn("EV / $1", format="$%.3f"),
                    "parlay_price": st.column_config.NumberColumn("Parlay odds"),
                },
            )

        st.markdown("### HR props workflow")
        st.write(
            "Use the HR Props tab for game-level dinger prices. Moneyline scoring now has real MLB factors; "
            "the next prop-specific upgrade is adding batter power and opposing-pitcher HR risk to the HR board."
        )

with ml_tab:
    st.subheader("Moneyline board")
    st.write(model_note)
    if plays_df.empty:
        st.info("No moneyline plays found with the current filters.")
    else:
        display_cols = [
            "commence_time", "game", "play", "best_book", "fair_prob", "market_fair_prob", "model_prob_delta",
            "fair_american", "best_implied_prob", "edge_pct", "ev_per_$1", "playable_to_label",
            "grade", "recommendation", "factor_summary"
        ]
        display = plays_df[[c for c in display_cols if c in plays_df.columns]].copy()
        display["commence_time"] = to_et(display["commence_time"])
        st.dataframe(
            display,
            use_container_width=True,
            column_config={
                "commence_time": st.column_config.DatetimeColumn("Start ET"),
                "fair_prob": st.column_config.NumberColumn("Model win %", format="%.1%"),
                "market_fair_prob": st.column_config.NumberColumn("Market win %", format="%.1%"),
                "model_prob_delta": st.column_config.NumberColumn("MLB factor +/-", format="%.2%"),
                "best_implied_prob": st.column_config.NumberColumn("Best implied %", format="%.1%"),
                "edge_pct": st.column_config.NumberColumn("Edge", format="%.2%"),
                "ev_per_$1": st.column_config.NumberColumn("EV / $1", format="$%.3f"),
                "fair_american": st.column_config.NumberColumn("Fair line"),
                "playable_to_label": "Playable to",
            },
            hide_index=True,
        )

        st.markdown("### Add a pick to tracker")
        pick_label = st.selectbox("Pick", plays_df["play"].tolist(), key="ml_pick")
        selected = plays_df.loc[plays_df["play"] == pick_label].iloc[0]
        stake = st.number_input("Stake", min_value=0.0, value=10.0, step=1.0, key="ml_stake")
        notes = st.text_input(
            "Notes",
            value=f"Grade {selected['grade']}; edge {selected['edge_pct']:.2%}; playable to {selected['playable_to_label']}",
            key="ml_notes",
        )
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

with factors_tab:
    st.subheader("Model factors")
    st.write(
        "This shows the baseball inputs that nudge the no-vig market probability. "
        "The market remains the anchor; these factors are intentionally modest adjustments."
    )
    if model_mode == "Market only":
        st.info("Model is set to Market only. Switch to 'Market + MLB factors' in the sidebar to use this tab.")
    elif features_df.empty:
        st.info("No MLB factor rows loaded. This can happen if the MLB stats endpoint is unavailable or the slate is from sample data.")
    else:
        factor_show = features_df.copy()
        if "commence_time" in factor_show.columns:
            factor_show["commence_time_et"] = to_et(factor_show["commence_time"])
        factor_cols = [
            "commence_time_et",
            "game",
            "team",
            "is_home",
            "venue",
            "official_status",
            "probable_pitcher",
            "opponent_probable_pitcher",
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
            "factor_summary",
        ]
        st.dataframe(
            factor_show[[c for c in factor_cols if c in factor_show.columns]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "commence_time_et": st.column_config.DatetimeColumn("Start ET"),
                "team_win_pct": st.column_config.NumberColumn("Win %", format="%.1%"),
                "opp_team_win_pct": st.column_config.NumberColumn("Opp win %", format="%.1%"),
                "record_adj": st.column_config.NumberColumn("Record adj", format="%.2%"),
                "offense_adj": st.column_config.NumberColumn("Offense adj", format="%.2%"),
                "team_pitching_adj": st.column_config.NumberColumn("Team pitching adj", format="%.2%"),
                "starter_adj": st.column_config.NumberColumn("Starter adj", format="%.2%"),
                "home_adj": st.column_config.NumberColumn("Home adj", format="%.2%"),
                "factor_adjustment": st.column_config.NumberColumn("Raw factor adj", format="%.2%"),
                "factor_confidence": st.column_config.NumberColumn("Data confidence", format="%.0%"),
            },
        )
        st.caption(
            "Positive adjustments help the listed team; negative adjustments hurt it. "
            "The final Today’s Card probability uses raw factor adjustment × data confidence × sidebar influence."
        )

with hr_tab:
    st.subheader("HR props")
    st.write(
        "This tab pulls `batter_home_runs` for one selected game and ranks the best available prices. "
        "It is still prop line-shopping; use the Model Factors tab to see probable pitcher context for each game."
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
