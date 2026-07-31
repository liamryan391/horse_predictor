from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

from racing_storage import RACE_COLUMNS, TABLES, seed_database_from_samples, write_races

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class APIConfig:
    provider: str
    base_url: str
    api_key: str = ""
    username: str = ""
    password: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch horse racing data, store it in SQL, and optionally export CSV snapshots."
    )
    parser.add_argument(
        "--provider",
        choices=["sample", "generic", "theracingapi", "ourhub"],
        default=os.getenv("HORSE_API_PROVIDER", "sample"),
        help="sample keeps the app usable without credentials; generic expects JSON matching the app schema.",
    )
    parser.add_argument("--base-url", default=os.getenv("HORSE_API_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("HORSE_API_KEY", ""))
    parser.add_argument("--username", default=os.getenv("RACING_API_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("RACING_API_PASSWORD", ""))
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL") or os.getenv("HORSE_DB_PATH") or str(BASE_DIR / "horse_racing.db"),
        help="SQLAlchemy database URL. Use mysql+pymysql://user:pass@host:3306/horse_predictor for MySQL.",
    )
    parser.add_argument("--historical-out", default=str(BASE_DIR / "sample_historical_data.csv"))
    parser.add_argument("--current-out", default=str(BASE_DIR / "sample_current_races.csv"))
    parser.add_argument("--days-ahead", type=int, default=7)
    parser.add_argument("--history-start", default=(date.today() - timedelta(days=365)).isoformat())
    parser.add_argument("--history-end", default=date.today().isoformat())
    parser.add_argument("--no-csv", action="store_true", help="Only update SQL; do not write CSV snapshots.")
    parser.add_argument("--repeat-hourly", action="store_true", help="Keep the ingestion worker running hourly.")
    return parser.parse_args()


def fetch_json(config: APIConfig, endpoint: str, params: Optional[dict] = None) -> dict | list:
    url = f"{config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {}
    auth = None

    if config.provider == "theracingapi":
        auth = (config.username, config.password)
    elif config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
        headers["X-API-Key"] = config.api_key

    response = requests.get(url, headers=headers, auth=auth, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_records(payload: dict | list, preferred_keys: Iterable[str]) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    for key in ["results", "racecards", "data"]:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def first_value(source: dict, *keys: str):
    for key in keys:
        value = source.get(key)
        if value not in (None, ""):
            if isinstance(value, dict):
                return first_value(value, "name", "title", "value")
            return value
    return None


def parse_distance(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    text = str(value).lower().replace(" ", "")
    miles = furlongs = yards = 0.0
    if "m" in text:
        head, text = text.split("m", 1)
        miles = float(head or 0)
    if "f" in text:
        head, text = text.split("f", 1)
        furlongs = float(head or 0)
    if "y" in text:
        head = text.split("y", 1)[0]
        yards = float(head or 0)
    parsed = miles * 1760 + furlongs * 220 + yards
    return parsed or None


def parse_odds(runner: dict):
    value = first_value(runner, "odds", "decimal_odds", "sp_dec", "sp")
    if isinstance(value, list) and value:
        value = first_value(value[0], "decimal", "odds", "price")
    if isinstance(value, dict):
        value = first_value(value, "decimal", "odds", "price")
    return value


def parse_class_rating(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return float(digits) if digits else None


def normalize_provider_records(records: Iterable[dict], current_mode: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for column in RACE_COLUMNS:
        if column not in df.columns:
            df[column] = None
    if current_mode:
        df["finishing_position"] = df["finishing_position"].fillna(0)
    return df[RACE_COLUMNS]


def flatten_racing_api_payload(payload: dict | list, current_mode: bool = False) -> pd.DataFrame:
    races = extract_records(payload, ["racecards", "results"])
    rows = []

    for race in races:
        if not isinstance(race, dict):
            continue
        runners = first_value(race, "runners", "horses")
        if not isinstance(runners, list) or not runners:
            runners = [race]

        for runner in runners:
            if not isinstance(runner, dict):
                continue
            row = {
                "race_date": first_value(race, "date", "race_date", "off_date"),
                "track": first_value(race, "course", "course_name", "track"),
                "distance": parse_distance(first_value(race, "distance_y", "distance", "race_distance")),
                "surface": first_value(race, "surface", "type", "race_type"),
                "horse": first_value(runner, "horse", "horse_name", "name"),
                "jockey": first_value(runner, "jockey", "jockey_name"),
                "owner": first_value(runner, "owner", "owner_name"),
                "trainer": first_value(runner, "trainer", "trainer_name"),
                "odds": parse_odds(runner),
                "finishing_position": first_value(runner, "position", "finishing_position", "pos"),
                "horse_age": first_value(runner, "age", "horse_age"),
                "horse_weight": first_value(runner, "weight", "horse_weight", "lbs"),
                "draw": first_value(runner, "draw", "stall"),
                "speed_rating": first_value(runner, "speed_rating"),
                "class_rating": first_value(runner, "class_rating"),
                "days_since_last_run": first_value(runner, "days_since_last_run", "last_run_days"),
                "past_bets_count": first_value(runner, "past_bets_count"),
                "past_bets_profit": first_value(runner, "past_bets_profit"),
                "weather": first_value(race, "weather", "going"),
            }
            rows.append(row)

    return normalize_provider_records(rows, current_mode=current_mode)


def expand_track_payload(payload: dict | list) -> list[tuple[str | None, dict]]:
    rows = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                rows.append((first_value(item, "track", "course"), item))
        return rows

    if not isinstance(payload, dict):
        return rows

    for track, races in payload.items():
        if isinstance(races, list):
            for race in races:
                if isinstance(race, dict):
                    rows.append((track, race))
        elif isinstance(races, dict):
            rows.append((track, races))
    return rows


def flatten_ourhub_payload(course_payload: dict | list, runner_payload: dict | list, race_date: str) -> pd.DataFrame:
    course_lookup = {}
    for track, race in expand_track_payload(course_payload):
        course_lookup[(track, first_value(race, "race_time", "off_time", "time"))] = race

    rows = []
    for track, race in expand_track_payload(runner_payload):
        race_time = first_value(race, "race_time", "off_time", "time")
        course = course_lookup.get((track, race_time), {})
        runners = first_value(race, "runners", "horses")
        if not isinstance(runners, list) or not runners:
            runners = [race]

        for runner in runners:
            if not isinstance(runner, dict):
                continue
            rows.append(
                {
                    "race_date": race_date,
                    "track": track or first_value(race, "course", "track"),
                    "distance": parse_distance(first_value(course, "distance") or first_value(race, "distance")),
                    "surface": first_value(course, "going") or first_value(race, "going"),
                    "horse": first_value(runner, "horse", "horse_name", "name"),
                    "jockey": first_value(runner, "jockey", "jockey_name"),
                    "owner": first_value(runner, "owner", "owner_name"),
                    "trainer": first_value(runner, "trainer", "trainer_name"),
                    "odds": parse_odds(runner),
                    "finishing_position": 0,
                    "horse_age": first_value(runner, "age", "horse_age"),
                    "horse_weight": first_value(runner, "weight", "horse_weight"),
                    "draw": first_value(runner, "draw", "stall", "number"),
                    "speed_rating": first_value(runner, "speed_rating"),
                    "class_rating": parse_class_rating(first_value(course, "race_class") or first_value(race, "race_class")),
                    "days_since_last_run": first_value(runner, "days_since_last_run"),
                    "past_bets_count": None,
                    "past_bets_profit": None,
                    "weather": first_value(course, "going") or first_value(race, "going"),
                }
            )

    return normalize_provider_records(rows, current_mode=True)


def fetch_generic(config: APIConfig, days_ahead: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.base_url or not config.api_key:
        raise SystemExit("Set HORSE_API_BASE_URL and HORSE_API_KEY for the generic provider.")

    historical_payload = fetch_json(config, "/historical-races")
    current_payload = fetch_json(config, "/current-races", params={"days_ahead": days_ahead})

    historical_df = normalize_provider_records(extract_records(historical_payload, ["historical", "results"]))
    current_df = normalize_provider_records(
        extract_records(current_payload, ["current", "racecards", "results"]),
        current_mode=True,
    )
    return historical_df, current_df


def fetch_ourhub(config: APIConfig, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.api_key:
        raise SystemExit("Set HORSE_API_KEY for OurHub Racing.")
    if not config.base_url:
        config.base_url = "https://api.ourhub.site/api"

    target_date = (date.today() + timedelta(days=args.days_ahead)).isoformat()
    course_payload = fetch_json(config, f"/course-info/{target_date}")
    runner_payload = fetch_json(config, f"/runner-info/{target_date}")
    return pd.DataFrame(columns=RACE_COLUMNS), flatten_ourhub_payload(course_payload, runner_payload, target_date)


def fetch_the_racing_api(config: APIConfig, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.username or not config.password:
        raise SystemExit("Set RACING_API_USERNAME and RACING_API_PASSWORD for The Racing API.")

    if not config.base_url:
        config.base_url = "https://api.theracingapi.com"

    historical_payload = fetch_json(
        config,
        "/v1/results",
        params={"start_date": args.history_start, "end_date": args.history_end, "limit": 50},
    )
    current_payload = fetch_json(config, "/v1/racecards", params={"day": "today"})
    return (
        flatten_racing_api_payload(historical_payload),
        flatten_racing_api_payload(current_payload, current_mode=True),
    )


def export_snapshots(historical_df: pd.DataFrame, current_df: pd.DataFrame, args: argparse.Namespace) -> None:
    if args.no_csv:
        return
    if not historical_df.empty:
        historical_df.to_csv(args.historical_out, index=False)
    if not current_df.empty:
        current_df.to_csv(args.current_out, index=False)


def run_once(args: argparse.Namespace) -> dict:
    if args.provider == "sample":
        counts = seed_database_from_samples(args.database_url, args.historical_out, args.current_out)
        return {"historical": counts["historical"], "current": counts["current"], "source": "sample"}

    config = APIConfig(
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        username=args.username,
        password=args.password,
    )

    if args.provider == "theracingapi":
        historical_df, current_df = fetch_the_racing_api(config, args)
        source = "theracingapi"
    elif args.provider == "ourhub":
        historical_df, current_df = fetch_ourhub(config, args)
        source = "ourhub"
    else:
        historical_df, current_df = fetch_generic(config, args.days_ahead)
        source = "generic"

    export_snapshots(historical_df, current_df, args)
    historical_count = 0
    if not historical_df.empty:
        historical_count = write_races(args.database_url, TABLES["historical"], historical_df, source=source)
    current_count = 0
    if not current_df.empty:
        current_count = write_races(args.database_url, TABLES["current"], current_df, source=source)
    return {"historical": historical_count, "current": current_count, "source": source}


def main() -> None:
    args = parse_args()
    while True:
        result = run_once(args)
        print(
            f"Updated {args.database_url}: {result['historical']} historical rows, "
            f"{result['current']} current rows from {result['source']}."
        )
        if not args.repeat_hourly:
            break
        time.sleep(60 * 60)


if __name__ == "__main__":
    main()
