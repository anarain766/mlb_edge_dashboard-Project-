# MLB Edge Dashboard

A Streamlit MVP for daily MLB betting decision support:

- Pulls live MLB moneyline odds from The Odds API
- Calculates implied probability
- Removes two-way vig to estimate a market-consensus fair probability
- Compares the best available sportsbook price to consensus
- Ranks moneyline plays by estimated edge and expected value
- Pulls event-level player props like `batter_home_runs` when an API key is available
- Builds simple 2-leg or 3-leg ML parlays while avoiding duplicate games
- Tracks bets, results, profit/loss, and ROI in SQLite

## Important note

This first version identifies **market value and line-shopping opportunities**. It is not yet a fully trained baseball prediction model. The next version should add pitcher, batter, park, weather, lineup, and injury features.

## Data source

This MVP uses The Odds API.

- Featured MLB markets such as moneyline use the `/sports/baseball_mlb/odds` endpoint.
- Player props such as `batter_home_runs` use the event-odds endpoint for a specific game.

## Setup

```bash
cd mlb_edge_dashboard
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

Create a secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`:

```toml
ODDS_API_KEY = "your-real-key-here"
```

Run the app:

```bash
streamlit run app.py
```

If you do not add an API key, the app loads sample data so you can test the layout.

## How the ML edge calculation works

For each game and sportsbook:

1. Convert both moneyline prices to raw implied probabilities.
2. Remove the book vig by dividing each side's raw implied probability by the two-team total.
3. Take the median no-vig probability across books as the market-consensus fair probability.
4. Find the best available price for each team.
5. Compare consensus fair probability to the best price's implied probability.
6. Estimate expected value per $1 staked.

This is useful because it finds cases where one sportsbook is offering a better price than the broader market.

## Next upgrades

Recommended build order:

1. Add probable pitcher features.
2. Add rolling team offense and bullpen form.
3. Add Baseball Savant / pybaseball batting and pitching features.
4. Add park factors and weather.
5. Train a calibrated moneyline model.
6. Train a separate HR probability model.
7. Add notifications before afternoon and evening slates.
8. Store closing line value so you can tell if the process is beating the market.

## Responsible use

This is a decision-support tool, not a guarantee of profit. Parlays and HR props are high variance. Track every pick and keep bet sizing small.
