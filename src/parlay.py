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
st.caption("Fast-loading daily moneyline grades, optional MLB-factor scoring, HR prop line-shopping, parlay ideas, and bet tracking.")


ET = "America/New_York"
GRADE_ORDER = {"D": 0, "Lean": 1, "C": 2, "B": 3, "A": 4}

PERCENT_COLUMNS = {
    "fair_prob",
    "market_fair_prob",
    "model_prob_delta",
    "best_implied_prob",
    "edge_pct",
    "estimated_hit_prob",
    "raw_implied_prob",
    "team_win_pct",
    "opp_team_win_pct",
    "record_adj",
    "offense_adj",
    "team_pitching_adj",
    "starter_adj",
    "home_adj",
    "factor_adjustment",
    "factor_confidence",
}


def pct_display_frame(df: pd.DataFrame, columns: set[str] | list[str] | None = None) -> pd.DataFrame:
    """Return a display-only copy where probability decimals are scaled to percentage points.

    Streamlit NumberColumn uses printf-style formatting, so 0.5427 with %.2f%% would
    display as 0.54%. Scaling to 54.27 first displays the intended 54.27%.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    target_cols = set(columns or PERCENT_COLUMNS)
    for col in target_cols.intersection(out.columns):
        out[col] = pd.to_numeric(out[col], errors="coerce") * 100
    return out


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
    grade = row.get("grade", "D")
    edge = row.get("edge_pct", 0)
    if grade in {"A", "B"}:
        return "Play"
    if grade == "C":
        return "Small play"
    if grade == "Lean":
        return "Lean only"
    if pd.notna(edge) and edge > 0:
        return "Early watch"
    return "Early grade"


def refresh_status_for_time(commence_time) -> str:
    """Label whether this is an early read or closer-to-first-pitch refresh."""
    game_time = pd.to_datetime(commence_time, errors="coerce", utc=True)
    if pd.isna(game_time):
        return "Refresh later"
    game_time_et = game_time.tz_convert(ET)
    now_et = pd.Timestamp.now(tz=ET)
    hours_to_first_pitch = (game_time_et - now_et).total_seconds() / 3600
    if hours_to_first_pitch <= -0.25:
        return "Started/live"
    if hours_to_first_pitch <= 1.25:
        return "Final check"
    if hours_to_first_pitch <= 3.5:
        return "Lineup/weather refresh"
    return "Early grade"


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def fmt_ev(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):.3f} per $1"


def describe_grade(row: pd.Series) -> str:
    grade = str(row.get("grade", "D"))
    refresh_status = str(row.get("refresh_status", "Early grade"))
    if grade in {"A", "B"}:
        return "This is one of the stronger current ML plays on the board."
    if grade == "C":
        return "This is playable, but I would size it smaller than an A/B spot."
    if grade == "Lean":
        return "This is more of a lean than a full play, so I would be price-sensitive."
    if refresh_status in {"Early grade", "Lineup/weather refresh"}:
        return "This is an early monitor grade, not a bet I would force yet."
    return "This is a lower-confidence candidate unless the number improves."


def describe_factor_read(row: pd.Series) -> str:
    delta = row.get("model_prob_delta", 0)
    summary = row.get("factor_summary", "")
    if delta is None or pd.isna(delta) or abs(float(delta)) < 0.0025:
        if summary and str(summary) != "Market only":
            return f"The MLB factors are mostly neutral here. {summary}"
        return "The current read is mostly market-based, so I would refresh later before locking it in."
    direction = "helping" if float(delta) > 0 else "hurting"
    return f"The MLB factor layer is {direction} this side by {fmt_pct(delta)}. {summary}"


def opinion_action(row: pd.Series) -> str:
    grade = str(row.get("grade", "D"))
    edge = row.get("edge_pct", 0)
    refresh_status = str(row.get("refresh_status", "Early grade"))
    if grade == "A":
        return "Top ML play"
    if grade == "B":
        return "Strong ML play"
    if grade == "C":
        return "Small single"
    if grade == "Lean":
        return "Lean / price watch"
    if pd.notna(edge) and float(edge) > 0:
        return "D monitor"
    if refresh_status in {"Early grade", "Lineup/weather refresh"}:
        return "D early grade"
    return "D low priority"


def suggested_unit(row: pd.Series) -> str:
    grade = str(row.get("grade", "D"))
    if grade == "A":
        return "0.75–1.00u"
    if grade == "B":
        return "0.50–0.75u"
    if grade == "C":
        return "0.25–0.50u"
    if grade == "Lean":
        return "0.25u max"
    return "monitor only"


def opinion_pick_sentence(row: pd.Series, rank: int) -> str:
    play = row.get("play", "")
    game = row.get("game", "")
    book = row.get("best_book", "")
    grade = row.get("grade", "D")
    action = row.get("opinion_action", opinion_action(row))
    rec = row.get("recommendation", "Early grade")
    unit = row.get("suggested_unit", suggested_unit(row))
    edge = fmt_pct(row.get("edge_pct"))
    model_win = fmt_pct(row.get("fair_prob"))
    market_win = fmt_pct(row.get("market_fair_prob", row.get("fair_prob")))
    implied = fmt_pct(row.get("best_implied_prob"))
    ev = fmt_ev(row.get("ev_per_$1"))
    playable_to = row.get("playable_to_label", "") or "n/a"
    refresh = row.get("refresh_status", "Refresh later")
    factor_read = describe_factor_read(row)

    if rank == 1:
        opener = "This is my favorite ML side on the current board"
    elif rank == 2:
        opener = "This is my second-best ML read right now"
    elif grade in {"A", "B", "C"}:
        opener = "This is still on my playable list"
    else:
        opener = "This is a watchlist side rather than something I would force"

    return (
        f"**{rank}. {play} — {action}.** {opener}. Game: {game}. Best book: **{book}**. "
        f"The model has it at **{model_win}** versus **{implied}** implied at the best current price "
        f"and **{market_win}** from the market baseline. That gives an estimated edge of **{edge}** "
        f"and EV of **{ev}**. Grade: **{grade}** / Recommendation: **{rec}** / Suggested sizing: **{unit}**. "
        f"I would only play this at **{playable_to} or better**. {factor_read} Refresh status: **{refresh}**."
    )


def build_daily_opinion_card(card_df: pd.DataFrame, max_picks: int = 6) -> tuple[str, list[str], list[str]]:
    """Create a decision-ready opinion card from the current board.

    This is meant to replace the daily manual question of "what are your best ML picks?"
    inside the app. It is deterministic from the app's current odds/model data, not a live LLM call.
    """
    if card_df.empty:
        return (
            "No ML candidates are loaded yet. Refresh once today's odds are available.",
            [],
            ["No parlay opinion yet because there are no ranked ML candidates loaded."],
        )

    rows = card_df.head(max_picks).copy()
    playable = card_df[card_df["grade"].isin(["A", "B", "C"])].copy()
    leans = card_df[card_df["grade"].isin(["Lean", "D"])].copy()

    if not playable.empty:
        favorite = playable.iloc[0]
        summary = (
            f"**My current ML card starts with {favorite.get('play', '')}.** "
            f"There are **{len(playable)}** A/B/C playable candidates and "
            f"**{int(card_df['grade'].isin(['Lean']).sum())}** lean/watch candidates. "
            "I would still keep early plays smaller until lineups, weather, and late market movement are checked. "
            "Singles should take priority over parlays."
        )
    else:
        favorite = card_df.iloc[0]
        summary = (
            f"**No full A/B/C ML play is showing yet, but my top early monitor is {favorite.get('play', '')}.** "
            "This is exactly the early-card use case: rank the board now, then refresh later before placing stronger bets. "
            "I would not force a full-unit ML until a later refresh improves the grade or confirms the setup."
        )

    pick_notes = [opinion_pick_sentence(row, i) for i, (_, row) in enumerate(rows.iterrows(), start=1)]

    parlay_notes: list[str] = []
    parlay_pool = card_df[card_df["grade"].isin(["A", "B", "C", "Lean"])].head(8)
    if len(parlay_pool) >= 2:
        two_leg = build_parlays(parlay_pool, legs=2, max_rows=8, max_combos=100).head(1)
        if not two_leg.empty:
            p = two_leg.iloc[0]
            parlay_notes.append(
                f"**Small parlay lean:** {p.get('plays', '')} at approximately **{format_american(p.get('parlay_price'))}**. "
                f"Estimated hit rate: **{fmt_pct(p.get('estimated_hit_prob'))}**. I would treat this as a small sprinkle only."
            )
    else:
        parlay_notes.append("No parlay lean yet. I want at least two Lean-or-better sides before building even a small 2-leg.")

    if not leans.empty:
        watch_names = ", ".join(leans.head(3)["play"].tolist())
        parlay_notes.append(f"**Refresh watchlist:** {watch_names}. These are the first sides I would re-check later in the day.")

    return summary, pick_notes, parlay_notes


def build_opinion_writeups(card_df: pd.DataFrame, max_picks: int = 5) -> list[str]:
    """Create opinion-style ML writeups from the current model rows.

    This is deterministic text from the current board. It does not make extra API
    calls, so it keeps the Today’s Card fast while giving a more practical read.
    """
    if card_df.empty:
        return []
    rows = card_df.head(max_picks).copy()
    writeups: list[str] = []
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        play = row.get("play", "")
        book = row.get("best_book", "")
        game = row.get("game", "")
        edge = fmt_pct(row.get("edge_pct"))
        model_win = fmt_pct(row.get("fair_prob"))
        market_win = fmt_pct(row.get("market_fair_prob", row.get("fair_prob")))
        implied = fmt_pct(row.get("best_implied_prob"))
        ev = fmt_ev(row.get("ev_per_$1"))
        playable_to = row.get("playable_to_label", "") or "n/a"
        grade = row.get("grade", "D")
        recommendation = row.get("recommendation", "Early grade")
        refresh_status = row.get("refresh_status", "Refresh later")
        grade_read = describe_grade(row)
        factor_read = describe_factor_read(row)

        if i == 1:
            lead = "My top current ML read"
        elif i == 2:
            lead = "Next-best ML read"
        else:
            lead = "Additional ML lean"

        writeups.append(
            f"**{i}. {play} — {lead}.** Game: {game}. Best price is at **{book}**. "
            f"The model has this at **{model_win}** versus **{implied}** implied at the best price "
            f"for an estimated edge of **{edge}** and EV of **{ev}**. "
            f"Grade: **{grade}** / Recommendation: **{recommendation}**. "
            f"I would only play it at **{playable_to} or better**. {grade_read} {factor_read} "
            f"Refresh status: **{refresh_status}**."
        )
    return writeups


def build_card_summary(card_df: pd.DataFrame) -> str:
    if card_df.empty:
        return "No ML candidates are loaded yet. Refresh the board after odds post."
    strong = int(card_df["grade"].isin(["A", "B"]).sum()) if "grade" in card_df.columns else 0
    playable = int(card_df["grade"].isin(["A", "B", "C", "Lean"]).sum()) if "grade" in card_df.columns else 0
    d_count = int((card_df["grade"] == "D").sum()) if "grade" in card_df.columns else 0
    best = card_df.iloc[0]
    return (
        f"Current board read: **{playable}** ML candidates are graded Lean or better, "
        f"including **{strong}** A/B plays. There are **{d_count}** D-grade early monitors. "
        f"The top-ranked side right now is **{best.get('play', '')}**, but the playable-to price still matters."
    )

def add_card_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["grade"] = out["grade"].fillna("D").replace({"No play": "D"})
    out["grade_rank"] = out["grade"].map(GRADE_ORDER).fillna(0)
    out["playable_to"] = out["fair_prob"].apply(playable_to_price)
    out["playable_to_label"] = out["playable_to"].apply(format_american)
    out["recommendation"] = out.apply(make_recommendation, axis=1)
    out["refresh_status"] = out["commence_time"].apply(refresh_status_for_time)
    out["opinion_action"] = out.apply(opinion_action, axis=1)
    out["suggested_unit"] = out.apply(suggested_unit, axis=1)
    out["card_score"] = (
        out["grade_rank"].astype(float) * 100
        + out["edge_pct"].fillna(0).astype(float) * 1000
        + out["ev_per_$1"].fillna(0).astype(float) * 100
        + out["fair_prob"].fillna(0).astype(float) * 10
    )
    return out.sort_values(["grade_rank", "ev_per_$1", "edge_pct", "fair_prob"], ascending=False)


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
def get_mlb_factor_data(events: list[dict], season: int, include_pitcher_stats: bool):
    return build_mlb_feature_frame(events, season=season, include_pitcher_stats=include_pitcher_stats)


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
    min_grade = st.selectbox("Minimum grade", ["All", "D", "Lean", "C", "B", "A"], index=0)

    st.divider()
    st.subheader("Model")
    model_mode = st.selectbox(
        "Moneyline model",
        ["Market only (fast early grade)", "Market + MLB factors"],
        index=0,
        help="Use fast market-only grades first, then switch to MLB factors later for a deeper refresh.",
    )
    model_season = st.number_input("MLB stats season", min_value=2020, max_value=2035, value=datetime.now().year, step=1)
    include_pitcher_stats = False
    if model_mode == "Market + MLB factors":
        include_pitcher_stats = st.checkbox(
            "Deep starter stat lookup",
            value=False,
            help="Adds probable-pitcher ERA/WHIP API calls. More complete, but slower on first load.",
        )
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
model_note = "Fast early grade: no-vig market consensus compared to best available price. Switch to MLB factors later for a deeper refresh."

if model_mode == "Market + MLB factors" and not base_plays_df.empty:
    try:
        features_df = get_mlb_factor_data(events, int(model_season), bool(include_pitcher_stats))
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
        "This tab turns the odds board into an early betting card. "
        f"Current scoring: {model_note} "
        "Grades update when you refresh odds later in the day; early D grades are monitor/rankings, not automatic bets. "
        "Only play a pick if the price is still at or better than the playable-to number."
    )

    if plays_df.empty:
        st.info("No moneyline plays found with the current filters.")
    else:
        card_df = plays_df.copy()
        card_df["commence_time_et"] = to_et(card_df["commence_time"])
        playable_df = card_df[card_df["grade"].isin(["A", "B", "C", "Lean"])].copy()
        top_df = card_df.head(8)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Playable MLs", len(playable_df))
        c2.metric("A/B plays", int(card_df["grade"].isin(["A", "B"]).sum()))
        best_edge = card_df["edge_pct"].max() if not card_df.empty else 0
        c3.metric("Best edge", f"{best_edge:.2%}")
        next_start = card_df["commence_time_et"].min()
        c4.metric("Next game ET", next_start.strftime("%-I:%M %p") if pd.notna(next_start) else "")

        st.markdown("## My take — today's ML card")
        st.caption(
            "This is designed to answer the daily question: 'What are your best ML picks today?' "
            "It updates from the app's current odds/model data whenever you refresh. It is opinion-style scoring, not a guarantee."
        )
        if st.button("Refresh today's card now", key="today_refresh_button"):
            st.cache_data.clear()
            st.rerun()

        card_summary, pick_notes, parlay_notes = build_daily_opinion_card(card_df, max_picks=6)
        st.markdown(card_summary)

        st.markdown("### What I would do right now")
        for pick_note in pick_notes:
            st.markdown(pick_note)

        st.markdown("### Parlay / refresh notes")
        for note in parlay_notes:
            st.markdown(note)

        st.markdown("### Best ML picks — ranked card")
        if top_df.empty:
            st.info("No moneyline candidates loaded yet. Refresh odds after the slate updates.")
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
                "opinion_action",
                "suggested_unit",
                "recommendation",
                "refresh_status",
                "factor_summary",
            ]
            show = top_df[[c for c in show_cols if c in top_df.columns]].copy()
            show = pct_display_frame(show)
            st.dataframe(
                show,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "commence_time_et": st.column_config.DatetimeColumn("Start ET"),
                    "best_book": "Best book",
                    "fair_prob": st.column_config.NumberColumn("Model win %", format="%.2f%%"),
                    "market_fair_prob": st.column_config.NumberColumn("Market win %", format="%.2f%%"),
                    "model_prob_delta": st.column_config.NumberColumn("MLB factor +/-", format="%.2f%%"),
                    "best_implied_prob": st.column_config.NumberColumn("Best implied %", format="%.2f%%"),
                    "edge_pct": st.column_config.NumberColumn("Edge", format="%.2f%%"),
                    "ev_per_$1": st.column_config.NumberColumn("EV / $1", format="$%.3f"),
                    "playable_to_label": "Playable to",
                    "opinion_action": "My action",
                    "suggested_unit": "Sizing",
                },
            )

            st.caption(
                "Grade guide: A/B = strongest current plays, C = smaller single, Lean = watch/small only, "
                "D = early monitor grade. Refresh later for confirmed lineups/weather and market movement. "
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
            two_leg_display = pct_display_frame(two_leg, ["estimated_hit_prob"])
            st.dataframe(
                two_leg_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "estimated_hit_prob": st.column_config.NumberColumn("Estimated hit %", format="%.2f%%"),
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
        display = pct_display_frame(display)
        st.dataframe(
            display,
            use_container_width=True,
            column_config={
                "commence_time": st.column_config.DatetimeColumn("Start ET"),
                "fair_prob": st.column_config.NumberColumn("Model win %", format="%.2f%%"),
                "market_fair_prob": st.column_config.NumberColumn("Market win %", format="%.2f%%"),
                "model_prob_delta": st.column_config.NumberColumn("MLB factor +/-", format="%.2f%%"),
                "best_implied_prob": st.column_config.NumberColumn("Best implied %", format="%.2f%%"),
                "edge_pct": st.column_config.NumberColumn("Edge", format="%.2f%%"),
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
    if model_mode == "Market only (fast early grade)":
        st.info("Model is set to Market only. Switch to 'Market + MLB factors' in the sidebar to use this tab.")
    elif features_df.empty:
        st.info("No MLB factor rows loaded. This tab populates only when you switch the sidebar to Market + MLB factors.")
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
        factor_display = pct_display_frame(factor_show[[c for c in factor_cols if c in factor_show.columns]])
        st.dataframe(
            factor_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "commence_time_et": st.column_config.DatetimeColumn("Start ET"),
                "team_win_pct": st.column_config.NumberColumn("Win %", format="%.2f%%"),
                "opp_team_win_pct": st.column_config.NumberColumn("Opp win %", format="%.2f%%"),
                "record_adj": st.column_config.NumberColumn("Record adj", format="%.2f%%"),
                "offense_adj": st.column_config.NumberColumn("Offense adj", format="%.2f%%"),
                "team_pitching_adj": st.column_config.NumberColumn("Team pitching adj", format="%.2f%%"),
                "starter_adj": st.column_config.NumberColumn("Starter adj", format="%.2f%%"),
                "home_adj": st.column_config.NumberColumn("Home adj", format="%.2f%%"),
                "factor_adjustment": st.column_config.NumberColumn("Raw factor adj", format="%.2f%%"),
                "factor_confidence": st.column_config.NumberColumn("Data confidence", format="%.2f%%"),
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
                        best_props_display = pct_display_frame(
                            best_props[["game", "player", "outcome", "point", "best_price", "best_book", "raw_implied_prob", "play"]],
                            ["raw_implied_prob"],
                        )
                        st.dataframe(
                            best_props_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "raw_implied_prob": st.column_config.NumberColumn("Raw implied %", format="%.2f%%")
                            },
                        )
                except Exception as e:
                    st.error(f"Could not load props: {e}")

with parlay_tab:
    st.subheader("Parlay builder")
    st.write("Parlays are high variance. This builder avoids duplicate games and uses the top ranked ML edges. It now waits until you click Generate so the app loads faster.")
    legs = st.selectbox("Legs", list(range(2, 9)), index=0)
    candidate_grades = st.multiselect("Use grades", ["A", "B", "C", "Lean", "D"], default=["A", "B", "C", "Lean", "D"])
    max_candidates = st.slider(
        "Max candidate picks to combine",
        min_value=6,
        max_value=14,
        value=8,
        step=1,
        help="Higher numbers create more combinations. Keep this lower for 6-8 leg parlays.",
    )
    max_combos = st.slider(
        "Max combinations to evaluate",
        min_value=250,
        max_value=5000,
        value=1500,
        step=250,
        help="Safety cap that keeps 6-8 leg parlays from slowing down the app.",
    )
    if st.button("Generate parlay ideas"):
        parlay_candidates = plays_df[plays_df["grade"].isin(candidate_grades)] if not plays_df.empty else plays_df
        parlays = build_parlays(parlay_candidates, legs=legs, max_rows=max_candidates, max_combos=max_combos)
        if parlays.empty:
            st.info("No parlays available with the selected filters.")
        else:
            parlays_display = pct_display_frame(parlays.head(20), ["estimated_hit_prob"])
            st.dataframe(
                parlays_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "estimated_hit_prob": st.column_config.NumberColumn("Estimated hit %", format="%.2f%%"),
                    "ev_per_$1": st.column_config.NumberColumn("EV / $1", format="$%.3f"),
                    "parlay_price": st.column_config.NumberColumn("Parlay odds"),
                },
            )
    else:
        st.info("Choose your grades/leg count, then click Generate parlay ideas. This keeps the daily card fast on load.")

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
        c4.metric("ROI", f"{roi:.2%}")
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
        if st.button("Show raw JSON", key="show_raw_json"):
            st.json(events)
        else:
            st.caption("Raw JSON is hidden until requested to keep the app lighter on normal loads.")
