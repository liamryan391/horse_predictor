from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import pandas as pd
import requests

from race_enrichment import enrich_race_frame
from schema import RACE_COLUMNS


DEFAULT_PROVIDER_BASE_URLS = {
    "ourhub": "https://api.ourhub.site/api",
    "theracingapi": "https://api.theracingapi.com",
}


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


@dataclass(frozen=True)
class RawProviderPayload:
    resource: str
    endpoint: str
    payload: dict | list
    params: dict | None = None


@dataclass
class ProviderResult:
    historical: pd.DataFrame
    current: pd.DataFrame
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    raw_payloads: list[RawProviderPayload] = field(default_factory=list)


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
                    self._raise_for_status(response)
                self._raise_for_status(response)
                return response.json()
            except requests.RequestException:
                if attempt >= self.config.retry_attempts:
                    raise
                time.sleep(self.config.retry_backoff_seconds * attempt)

        raise RuntimeError(f"Failed to fetch {url}")

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip().replace("\n", " ")[:300]
            message = f"{exc}"
            if detail:
                message = f"{message}. Response detail: {detail}"
            raise requests.HTTPError(message, response=response) from exc

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
    return combine_payload_pages(fetch_paginated_payloads(client, endpoint, params), preferred_keys)


def fetch_paginated_payloads(client: HTTPClient, endpoint: str, params: dict) -> list[dict | list]:
    payloads = []
    next_page: int | None = 1
    page_count = 0
    while next_page is not None and page_count < client.config.max_pages:
        page_params = {**params, "page": next_page}
        payload = client.fetch_json(endpoint, params=page_params)
        payloads.append(payload)
        page_count += 1
        next_page = maybe_next_page(payload)
    return payloads


def has_http_base_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resolve_provider_base_url(provider: str, base_url: str = "", allow_override: bool = False) -> str:
    if allow_override and has_http_base_url(base_url):
        return base_url
    return DEFAULT_PROVIDER_BASE_URLS.get(provider, base_url)


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

        current_params = {"days_ahead": context.days_ahead}
        historical_payload = self.client.fetch_json("/historical-races")
        current_payload = self.client.fetch_json("/current-races", params=current_params)

        historical_df, historical_issues = normalize_provider_records(
            extract_records(historical_payload, ["historical", "results"]),
            current_mode=False,
        )
        current_df, current_issues = normalize_provider_records(
            extract_records(current_payload, ["current", "racecards", "results"]),
            current_mode=True,
        )
        return ProviderResult(
            historical_df,
            current_df,
            historical_issues + current_issues,
            raw_payloads=[
                RawProviderPayload("historical_races", "/historical-races", historical_payload),
                RawProviderPayload("current_races", "/current-races", current_payload, current_params),
            ],
        )


class OurHubProviderAdapter(ProviderAdapter):
    provider_name = "ourhub"

    def fetch(self, context: FetchContext) -> ProviderResult:
        if not self.config.api_key:
            raise ValueError("Set HORSE_API_KEY for OurHub Racing.")
        resolved_base_url = resolve_provider_base_url(self.provider_name, self.config.base_url)
        if self.config.base_url != resolved_base_url:
            self.config = APIConfig(**{**self.config.__dict__, "base_url": resolved_base_url})
            self.client = HTTPClient(self.config)

        target_date = (date.today() + timedelta(days=context.days_ahead)).isoformat()
        course_endpoint = f"/course-info/{target_date}"
        runner_endpoint = f"/runner-info/{target_date}"
        course_payload = self.client.fetch_json(course_endpoint)
        runner_payload = self.client.fetch_json(runner_endpoint)
        current_df, issues = flatten_ourhub_payload(course_payload, runner_payload, target_date)
        return ProviderResult(
            pd.DataFrame(columns=RACE_COLUMNS),
            current_df,
            issues,
            raw_payloads=[
                RawProviderPayload("course_info", course_endpoint, course_payload),
                RawProviderPayload("runner_info", runner_endpoint, runner_payload),
            ],
        )


class TheRacingApiProviderAdapter(ProviderAdapter):
    provider_name = "theracingapi"

    def fetch(self, context: FetchContext) -> ProviderResult:
        if not self.config.username or not self.config.password:
            raise ValueError("Set RACING_API_USERNAME and RACING_API_PASSWORD for The Racing API.")
        resolved_base_url = resolve_provider_base_url(self.provider_name, self.config.base_url)
        if self.config.base_url != resolved_base_url:
            self.config = APIConfig(**{**self.config.__dict__, "base_url": resolved_base_url})
            self.client = HTTPClient(self.config)

        historical_params = {"start_date": context.history_start, "end_date": context.history_end, "limit": 50}
        historical_payloads = fetch_paginated_payloads(
            self.client,
            "/v1/results",
            historical_params,
        )
        historical_records = combine_payload_pages(historical_payloads, ["results"])
        current_params = {"day": "today"}
        current_payload = self.client.fetch_json("/v1/racecards", params=current_params)

        historical_df, historical_issues = flatten_racing_api_records(historical_records, current_mode=False)
        current_df, current_issues = flatten_racing_api_payload(current_payload, current_mode=True)
        return ProviderResult(
            historical_df,
            current_df,
            historical_issues + current_issues,
            raw_payloads=[
                *[
                    RawProviderPayload("results", "/v1/results", payload, {**historical_params, "page": index + 1})
                    for index, payload in enumerate(historical_payloads)
                ],
                RawProviderPayload("racecards", "/v1/racecards", current_payload, current_params),
            ],
        )


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


def split_ourhub_race_label(label: str, course_names: Iterable[str]) -> tuple[str | None, str | None, str | None]:
    normalized_label = label.strip()
    for course_name in sorted((name for name in course_names if name), key=len, reverse=True):
        if normalized_label.lower().startswith(f"{course_name.lower()} "):
            remainder = normalized_label[len(course_name) :].strip()
            match = re.match(r"^(?P<time>\d{1,2}:\d{2})\s*(?P<name>.*)$", remainder)
            if match:
                return course_name, match.group("time"), match.group("name").strip() or None
            return course_name, None, remainder or None

    match = re.match(r"^(?P<course>.+?)\s+(?P<time>\d{1,2}:\d{2})\s*(?P<name>.*)$", normalized_label)
    if match:
        return match.group("course").strip(), match.group("time"), match.group("name").strip() or None
    return normalized_label or None, None, None


def expand_ourhub_runner_payload(payload: dict | list, course_names: Iterable[str]) -> list[tuple[str | None, dict]]:
    if not isinstance(payload, dict):
        return expand_track_payload(payload)

    rows = []
    for race_label, runners in payload.items():
        track, race_time, race_name = split_ourhub_race_label(str(race_label), course_names)
        race = {
            "race_time": race_time,
            "race_name": race_name,
            "runners": runners if isinstance(runners, list) else [],
        }
        rows.append((track, race))
    return rows


def flatten_ourhub_payload(
    course_payload: dict | list,
    runner_payload: dict | list,
    race_date: str,
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    course_lookup = {}
    course_names = list(course_payload.keys()) if isinstance(course_payload, dict) else []
    for track, race in expand_track_payload(course_payload):
        course_lookup[(track, first_value(race, "race_time", "off_time", "time"))] = race

    rows = []
    for track, race in expand_ourhub_runner_payload(runner_payload, course_names):
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
