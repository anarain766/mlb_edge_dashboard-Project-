# MLB Edge Dashboard

Streamlit app for daily MLB betting analysis: moneyline grades, best available prices, MLB-factor adjustments, opinion-style Today's Card, HR prop line-shopping, parlay ideas, and bet tracking.

## Latest update

The Today's Card now includes **My take — today's ML card**, designed to replace asking for a daily opinion manually. It generates:

- My current top ML read
- Ranked ML pick write-ups
- Bet/monitor action labels
- Suggested sizing bands
- Playable-to prices
- Parlay/refresh notes
- Refresh watchlist

This uses the current odds/model data already loaded in the app. Refresh later in the day to update line movement, odds, and grades.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud secret

Add this in Streamlit Cloud > App settings > Secrets:

```toml
ODDS_API_KEY = "your-real-api-key"
```

