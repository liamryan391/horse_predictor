import argparse
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd
import requests

HISTORICAL_COLUMNS = [
    "race_date",
    "track",
    "distance",
    "surface",
    "horse",
    "jockey",
    "owner",
    "trainer",
    "odds",
    "finishing_position",
    "horse_age",
    "horse_weight",
    "draw",
    "speed_rating",
    "class_rating",
    "days_since_last_run",
    "past_bets_count",
    "past_bets_profit",
    "weather",
]

CURRENT_COLUMNS = HISTORICAL_COLUMNS.copy()


@dataclass
class APIConfig:
    base_url: str
    api_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch horse racing data from API, store in SQL, and generate sample CSV files."
    )
    parser.add_argument("--base-url", default=os.getenv("HORSE_API_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("HORSE_API_KEY", ""))
    parser.add_argument("--db-path", default="horse_racing.db")
    parser.add_argument("--historical-out", default="sample_historical_data.csv")
    parser.add_argument("--current-out", default="sample_current_races.csv")
    parser.add_argument("--days-ahead", type=int, default=7)
    return parser.parse_args()


def fetch_json(config: APIConfig, endpoint: str, params: Optional[dict] = None) -> List[dict]:
    url = f"{config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Authorization": f"Bearer {config.api_key}"}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unsupported payload structure from endpoint '{endpoint}'")


def normalize_records(records: Iterable[dict], column_order: List[str], current_mode: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in column_order:
        if col not in df.columns:
            df[col] = None

    if current_mode:
        df["finishing_position"] = df["finishing_position"].fillna(0)

    return df[column_order]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS races_historical (
            race_date TEXT,
            track TEXT,
            distance REAL,
            surface TEXT,
            horse TEXT,
            jockey TEXT,
            owner TEXT,
            trainer TEXT,
            odds REAL,
            finishing_position INTEGER,
            horse_age REAL,
            horse_weight REAL,
            draw REAL,
            speed_rating REAL,
            class_rating REAL,
            days_since_last_run REAL,
            past_bets_count REAL,
            past_bets_profit REAL,
            weather TEXT,
            ingested_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS races_current (
            race_date TEXT,
            track TEXT,
            distance REAL,
            surface TEXT,
            horse TEXT,
            jockey TEXT,
            owner TEXT,
            trainer TEXT,
            odds REAL,
            finishing_position INTEGER,
            horse_age REAL,
            horse_weight REAL,
            draw REAL,
            speed_rating REAL,
            class_rating REAL,
            days_since_last_run REAL,
            past_bets_count REAL,
            past_bets_profit REAL,
            weather TEXT,
            ingested_at TEXT
        )
        """
    )
    conn.commit()


def write_table(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    write_df = df.copy()
    write_df["ingested_at"] = datetime.utcnow().isoformat()
    write_df.to_sql(table, conn, if_exists="replace", index=False)


def main() -> None:
    args = parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit(
            "Missing API config. Set HORSE_API_BASE_URL and HORSE_API_KEY or pass --base-url/--api-key."
        )

    config = APIConfig(base_url=args.base_url, api_key=args.api_key)

    historical_records = fetch_json(config, "/historical-races")
    current_records = fetch_json(config, "/current-races", params={"days_ahead": args.days_ahead})

    historical_df = normalize_records(historical_records, HISTORICAL_COLUMNS)
    current_df = normalize_records(current_records, CURRENT_COLUMNS, current_mode=True)

    historical_df.to_csv(args.historical_out, index=False)
    current_df.to_csv(args.current_out, index=False)

    with sqlite3.connect(args.db_path) as conn:
        init_db(conn)
        write_table(conn, "races_historical", historical_df)
        write_table(conn, "races_current", current_df)

    print(f"Saved {len(historical_df)} historical rows to {args.historical_out}")
    print(f"Saved {len(current_df)} current rows to {args.current_out}")
    print(f"Data also loaded into SQLite DB: {args.db_path}")


if __name__ == "__main__":
    main()
