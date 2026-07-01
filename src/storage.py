from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

DB_PATH = Path("data/bets.db")


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                event_date TEXT,
                sport TEXT DEFAULT 'MLB',
                bet_type TEXT,
                play TEXT,
                book TEXT,
                odds INTEGER,
                stake REAL,
                result TEXT DEFAULT 'pending',
                profit_loss REAL DEFAULT 0,
                notes TEXT
            )
            """
        )
        conn.commit()


def add_bet(event_date: str, bet_type: str, play: str, book: str, odds: int, stake: float, notes: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO bets(event_date, bet_type, play, book, odds, stake, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_date, bet_type, play, book, int(odds), float(stake), notes),
        )
        conn.commit()


def load_bets() -> pd.DataFrame:
    init_db()
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM bets ORDER BY created_at DESC", conn)


def update_result(bet_id: int, result: str, profit_loss: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE bets SET result = ?, profit_loss = ? WHERE id = ?",
            (result, float(profit_loss), int(bet_id)),
        )
        conn.commit()
