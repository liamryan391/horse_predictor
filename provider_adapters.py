from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, List, Optional

import pandas as pd
import requests

from race_enrichment import enrich_race_frame
from schema import RACE_COLUMNS


@dataclass(frozen=True)
class APIConfig:
    provider: str
    base_url: str
    api_key: str = ""
    username: str = ""
    password: str = ""
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.5
    min_request_interval_seconds: float = 0.0
    max_pages: int = 5


@dataclass(frozen=True)
class FetchContext:
    days_ahead: int
    history_start: str
    history_end: str


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    row_number: int
    field: str
    message: str


@dataclass
class ProviderResult:
    historical: pd.DataFrame
    current: pd.DataFrame
    validation_issues: list[ValidationIssue] = field(default_factory=list)


class HTTPClient:
    def __init__(self, config: APIConfig):
        self.config = config
        self._last_request_at = 0.0

    def fetch_json(self, endpoint: str, params: Optional[dict] = None) -> dict | list:
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {}
        auth = None

        if self.config.provider == "theracingapi":
            auth = (self.config.username, self.config.password)
        elif self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
            headers["X-API-Key"] = self.config.api_key

        for attempt in range(1, self.config.retry_attempts + 1):
            self._respect_rate_limit()
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    auth=auth,
                    params=params,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
                    response.raise_for_status()
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt >= self.config.retry_attempts:
                    raise
                time.sleep(self.config.retry_backoff_seconds * attempt)

        raise RuntimeError(f"Failed to fetch {url}")

    def _respect_rate_limit(self) -> None:
        if self.config.min_request_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        delay = self.config.min_request_interval_seconds - elapsed
        if delay > 0:
            time.sleep(delay)
        self._last_request_at = time.monotonic()


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
    value = None
    for key in ["odds", "decimal_odds", "sp_dec", "sp"]:
        candidate = runner.get(key)
        if candidate not in (None, ""):
            value = candidate
            break
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


def normalize_provider_records(records: Iterable[dict], current_mode: bool = False) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    df = pd.DataFrame(records)
    for column in RACE_COLUMNS:
        if column not in df.columns:
            df[column] = None
    if current_mode:
        df.loc[df["finishing_position"].isna(), "finishing_position"] = 0

    df = df[RACE_COLUMNS].copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.date
    for column in [
        "distance",
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
    ]:
        if column == "distance":
            df[column] = df[column].map(parse_distance)
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    issues: list[ValidationIssue] = []
    required = ["race_date", "track", "horse"]
    if not current_mode:
        required.append("finishing_position")

    invalid_mask = pd.Series(False, index=df.index)
    for field_name in required:
        missing = df[field_name].isna() | (df[field_name].astype(str).str.strip() == "")
        for row_number in df.index[missing].tolist():
            issues.append(
                ValidationIssue(
                    severity="error",
                    row_number=int(row_number),
                    field=field_name,
                    message="Required provider field is missing or invalid.",
                )
            )
        invalid_mask = invalid_mask | missing

    for field_name in ["distance", "surface", "jockey", "owner", "trainer", "odds"]:
        missing = df[field_name].isna() | (df[field_name].astype(str).str.strip() == "")
        for row_number in df.index[missing].tolist():
            issues.append(
                ValidationIssue(
                    severity="warning",
                    row_number=int(row_number),
                    field=field_name,
                    message="Recommended prediction field is missing or invalid.",
                )
            )

    return enrich_race_frame(df.loc[~invalid_mask].reset_index(drop=True)), issues


def summarize_validation_issues(issues: list[ValidationIssue]) -> str | None:
    if not issues:
        return None
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")
    fields = sorted({issue.field for issue in issues})
    return f"validation issues: {errors} errors, {warnings} warnings across {', '.join(fields)}"


def combine_payload_pages(payloads: list[dict | list], preferred_keys: Iterable[str]) -> list[dict]:
    records: list[dict] = []
    for payload in payloads:
        records.extend(extract_records(payload, preferred_keys))
    return records


def maybe_next_page(payload: dict | list) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ["next_page", "nextPage"]:
        value = payload.get(key)
        if isinstance(value, int):
            return value
    pagination = payload.get("pagination")
    if isinstance(pagination, dict):
        value = pagination.get("next_page") or pagination.get("nextPage")
        if isinstance(value, int):
            return value
    return None


def fetch_paginated(client: HTTPClient, endpoint: str, params: dict, preferred_keys: Iterable[str]) -> list[dict]:
    payloads = []
    next_page: int | None = 1
    page_count = 0
    while next_page is not None and page_count < client.config.max_pages:
        page_params = {**params, "page": next_page}
        payload = client.fetch_json(endpoint, params=page_params)
        payloads.append(payload)
        page_count += 1
        next_page = maybe_next_page(payload)
    return combine_payload_pages(payloads, preferred_keys)


class ProviderAdapter:
    provider_name = "base"

    def __init__(self, config: APIConfig):
        self.config = config
        self.client = HTTPClient(config)

    def fetch(self, context: FetchContext) -> ProviderResult:
        raise NotImplementedError


class GenericProviderAdapter(ProviderAdapter):
    provider_name = "generic"

    def fetch(self, context: FetchContext) -> ProviderResult:
        if not self.config.base_url or not self.config.api_key:
            raise ValueError("Set HORSE_API_BASE_URL and HORSE_API_KEY for the generic provider.")

        historical_payload = self.client.fetch_json("/historical-races")
        current_payload = self.client.fetch_json("/current-races", params={"days_ahead": context.days_ahead})

        historical_df, historical_issues = normalize_provider_records(
            extract_records(historical_payload, ["historical", "results"]),
            current_mode=False,
        )
        current_df, current_issues = normalize_provider_records(
            extract_records(current_payload, ["current", "racecards", "results"]),
            current_mode=True,
        )
        return ProviderResult(historical_df, current_df, historical_issues + current_issues)


class OurHubProviderAdapter(ProviderAdapter):
    provider_name = "ourhub"

    def fetch(self, context: FetchContext) -> ProviderResult:
        if not self.config.api_key:
            raise ValueError("Set HORSE_API_KEY for OurHub Racing.")
        if not self.config.base_url:
            self.config = APIConfig(**{**self.config.__dict__, "base_url": "https://api.ourhub.site/api"})
            self.client = HTTPClient(self.config)

        target_date = (date.today() + timedelta(days=context.days_ahead)).isoformat()
        course_payload = self.client.fetch_json(f"/course-info/{target_date}")
        runner_payload = self.client.fetch_json(f"/runner-info/{target_date}")
        current_df, issues = flatten_ourhub_payload(course_payload, runner_payload, target_date)
        return ProviderResult(pd.DataFrame(columns=RACE_COLUMNS), current_df, issues)


class TheRacingApiProviderAdapter(ProviderAdapter):
    provider_name = "theracingapi"

    def fetch(self, context: FetchContext) -> ProviderResult:
        if not self.config.username or not self.config.password:
            raise ValueError("Set RACING_API_USERNAME and RACING_API_PASSWORD for The Racing API.")
        if not self.config.base_url:
            self.config = APIConfig(**{**self.config.__dict__, "base_url": "https://api.theracingapi.com"})
            self.client = HTTPClient(self.config)

        historical_records = fetch_paginated(
            self.client,
            "/v1/results",
            {"start_date": context.history_start, "end_date": context.history_end, "limit": 50},
            ["results"],
        )
        current_payload = self.client.fetch_json("/v1/racecards", params={"day": "today"})

        historical_df, historical_issues = flatten_racing_api_records(historical_records, current_mode=False)
        current_df, current_issues = flatten_racing_api_payload(current_payload, current_mode=True)
        return ProviderResult(historical_df, current_df, historical_issues + current_issues)


def get_provider_adapter(config: APIConfig) -> ProviderAdapter:
    adapters = {
        GenericProviderAdapter.provider_name: GenericProviderAdapter,
        OurHubProviderAdapter.provider_name: OurHubProviderAdapter,
        TheRacingApiProviderAdapter.provider_name: TheRacingApiProviderAdapter,
    }
    try:
        return adapters[config.provider](config)
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {config.provider}") from exc


def flatten_racing_api_records(records: Iterable[dict], current_mode: bool = False) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    rows = []
    for race in records:
        if not isinstance(race, dict):
            continue
        runners = first_value(race, "runners", "horses")
        if not isinstance(runners, list) or not runners:
            runners = [race]

        for runner in runners:
            if not isinstance(runner, dict):
                continue
            rows.append(
                {
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
            )
    return normalize_provider_records(rows, current_mode=current_mode)


def flatten_racing_api_payload(payload: dict | list, current_mode: bool = False) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    return flatten_racing_api_records(extract_records(payload, ["racecards", "results"]), current_mode=current_mode)


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


def flatten_ourhub_payload(
    course_payload: dict | list,
    runner_payload: dict | list,
    race_date: str,
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
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
