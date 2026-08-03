from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provider_adapters import (  # noqa: E402
    APIConfig,
    FetchContext,
    HTTPClient,
    get_provider_adapter,
    resolve_provider_base_url,
    summarize_validation_issues,
)
from settings import get_settings  # noqa: E402


DEFAULT_COVERAGE_THRESHOLDS = {
    "ourhub": {
        "track": 1.0,
        "distance": 0.95,
        "surface": 0.95,
        "horse": 1.0,
        "jockey": 0.95,
        "trainer": 0.95,
    },
    "theracingapi": {
        "track": 1.0,
        "distance": 0.9,
        "surface": 0.75,
        "horse": 1.0,
        "jockey": 0.75,
        "trainer": 0.75,
        "odds": 0.75,
    },
    "generic": {
        "track": 1.0,
        "distance": 0.75,
        "horse": 1.0,
    },
}

DEFAULT_MIN_ROWS = {
    "ourhub": {"historical": 0, "current": 1},
    "theracingapi": {"historical": 1, "current": 1},
    "generic": {"historical": 1, "current": 1},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def credential_state(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    return {"present": bool(cleaned), "length": len(cleaned)}


def redacted_config(config: APIConfig) -> dict[str, Any]:
    return {
        "provider": config.provider,
        "baseUrl": config.base_url,
        "apiKey": credential_state(config.api_key),
        "username": credential_state(config.username),
        "password": credential_state(config.password),
        "timeoutSeconds": config.timeout_seconds,
        "retryAttempts": config.retry_attempts,
        "minRequestIntervalSeconds": config.min_request_interval_seconds,
        "maxPages": config.max_pages,
    }


def parse_field_thresholds(values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Coverage threshold must use FIELD=VALUE format: {value}")
        field, raw_threshold = value.split("=", 1)
        field = field.strip()
        threshold = float(raw_threshold)
        if not field:
            raise ValueError("Coverage threshold field must not be empty.")
        if not 0 <= threshold <= 1:
            raise ValueError(f"Coverage threshold for {field} must be between 0 and 1.")
        thresholds[field] = threshold
    return thresholds


def default_base_url(provider: str, base_url: str, allow_override: bool = False) -> str:
    return resolve_provider_base_url(provider, base_url, allow_override=allow_override)


def build_config(args: argparse.Namespace) -> APIConfig:
    settings = get_settings()
    return APIConfig(
        provider=args.provider,
        base_url=default_base_url(
            args.provider,
            args.base_url if args.base_url is not None else settings.horse_api_base_url,
            allow_override=args.allow_base_url_override,
        ),
        api_key=args.api_key if args.api_key is not None else settings.horse_api_key,
        username=args.username if args.username is not None else settings.racing_api_username,
        password=args.password if args.password is not None else settings.racing_api_password,
        timeout_seconds=args.timeout_seconds,
        retry_attempts=args.retry_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        min_request_interval_seconds=args.min_request_interval_seconds,
        max_pages=args.max_pages,
    )


def endpoint_rows(payload: dict | list) -> int:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return 0
    for key in ["results", "racecards", "data", "courses"]:
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    list_lengths = [len(value) for value in payload.values() if isinstance(value, list)]
    return sum(list_lengths) if list_lengths else len(payload)


def probe_endpoint(client: HTTPClient, name: str, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        payload = client.fetch_json(endpoint, params=params)
    except requests.HTTPError as exc:
        response = exc.response
        return {
            "name": name,
            "endpoint": endpoint,
            "status": "failure",
            "statusCode": response.status_code if response is not None else None,
            "rows": 0,
            "message": str(exc),
        }
    except requests.RequestException as exc:
        return {"name": name, "endpoint": endpoint, "status": "failure", "statusCode": None, "rows": 0, "message": str(exc)}

    return {"name": name, "endpoint": endpoint, "status": "success", "statusCode": 200, "rows": endpoint_rows(payload)}


def provider_endpoint_probes(config: APIConfig, context: FetchContext) -> list[dict[str, Any]]:
    client = HTTPClient(config)
    if config.provider == "ourhub":
        target_date = (date.today() + timedelta(days=context.days_ahead)).isoformat()
        return [
            probe_endpoint(client, "course_info", f"/course-info/{target_date}"),
            probe_endpoint(client, "runner_info", f"/runner-info/{target_date}"),
        ]
    if config.provider == "theracingapi":
        return [
            probe_endpoint(client, "courses", "/v1/courses"),
            probe_endpoint(client, "racecards", "/v1/racecards", {"day": "today"}),
            probe_endpoint(
                client,
                "results",
                "/v1/results",
                {"start_date": context.history_start, "end_date": context.history_end, "limit": 5, "page": 1},
            ),
        ]
    if config.provider == "generic":
        return [
            probe_endpoint(client, "historical_races", "/historical-races"),
            probe_endpoint(client, "current_races", "/current-races", {"days_ahead": context.days_ahead}),
        ]
    return []


def frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "tracks": 0, "trackSample": [], "coverage": {}}

    tracks = sorted(str(track) for track in frame["track"].dropna().unique().tolist()) if "track" in frame.columns else []
    coverage = {}
    for column in frame.columns:
        if column in {"race_date", "track", "distance", "surface", "horse", "jockey", "owner", "trainer", "odds", "draw", "horse_weight", "class_rating"}:
            series = frame[column]
            coverage[column] = round(float((series.notna() & (series.astype(str).str.strip() != "")).mean()), 3)
    return {"rows": int(len(frame)), "tracks": len(tracks), "trackSample": tracks[:12], "coverage": coverage}


def issue_counts(issues: list[Any]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0}
    for issue in issues:
        severity = getattr(issue, "severity", "warning")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def acceptance_failures(
    provider: str,
    historical_summary: dict[str, Any],
    current_summary: dict[str, Any],
    issues: list[Any],
    thresholds: dict[str, float],
    min_historical_rows: int,
    min_current_rows: int,
    max_error_issues: int,
    max_warning_issues: int | None,
) -> list[str]:
    failures = []
    if historical_summary["rows"] < min_historical_rows:
        failures.append(f"historical rows {historical_summary['rows']} below minimum {min_historical_rows}")
    if current_summary["rows"] < min_current_rows:
        failures.append(f"current rows {current_summary['rows']} below minimum {min_current_rows}")

    counts = issue_counts(issues)
    if counts.get("error", 0) > max_error_issues:
        failures.append(f"error validation issues {counts.get('error', 0)} above maximum {max_error_issues}")
    if max_warning_issues is not None and counts.get("warning", 0) > max_warning_issues:
        failures.append(f"warning validation issues {counts.get('warning', 0)} above maximum {max_warning_issues}")

    coverage = current_summary["coverage"] if current_summary["rows"] else historical_summary["coverage"]
    for field, threshold in thresholds.items():
        actual = coverage.get(field)
        if actual is None:
            failures.append(f"{field} coverage is missing for {provider}")
        elif actual < threshold:
            failures.append(f"{field} coverage {actual:.0%} below threshold {threshold:.0%}")
    return failures


def run_smoke(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    thresholds = {**DEFAULT_COVERAGE_THRESHOLDS.get(args.provider, {}), **parse_field_thresholds(args.min_field_coverage)}
    min_rows = DEFAULT_MIN_ROWS.get(args.provider, {"historical": 0, "current": 1})
    min_historical_rows = args.min_historical_rows if args.min_historical_rows is not None else min_rows["historical"]
    min_current_rows = args.min_current_rows if args.min_current_rows is not None else min_rows["current"]

    config = build_config(args)
    context = FetchContext(args.days_ahead, args.history_start, args.history_end)
    endpoints = provider_endpoint_probes(config, context) if args.probe_endpoints else []
    payload: dict[str, Any] = {
        "generatedAt": utc_now(),
        "provider": args.provider,
        "config": redacted_config(config),
        "endpoints": endpoints,
    }

    try:
        result = get_provider_adapter(config).fetch(context)
    except Exception as exc:
        payload["fetch"] = {"status": "failure", "message": str(exc)}
        payload["acceptance"] = {"status": "failed", "failures": [str(exc)]}
        return payload, 0 if args.allow_failure else 1

    historical_summary = frame_summary(result.historical)
    current_summary = frame_summary(result.current)
    failures = acceptance_failures(
        args.provider,
        historical_summary,
        current_summary,
        result.validation_issues,
        thresholds,
        min_historical_rows,
        min_current_rows,
        args.max_error_issues,
        args.max_warning_issues,
    )
    payload["fetch"] = {
        "status": "success",
        "historical": historical_summary,
        "current": current_summary,
        "validationIssues": len(result.validation_issues),
        "validationIssueCounts": issue_counts(result.validation_issues),
        "validationSummary": summarize_validation_issues(result.validation_issues),
    }
    payload["acceptance"] = {"status": "failed" if failures else "passed", "failures": failures}
    return payload, 1 if failures and not args.allow_failure else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run a redacted provider smoke check for Horse Predictor.")
    parser.add_argument("--provider", choices=["generic", "ourhub", "theracingapi"], default=settings.horse_api_provider)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--allow-base-url-override",
        action="store_true",
        help="Let --base-url or HORSE_API_BASE_URL override built-in provider hosts.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--days-ahead", type=int, default=0)
    parser.add_argument("--history-start", default=(date.today() - timedelta(days=7)).isoformat())
    parser.add_argument("--history-end", default=date.today().isoformat())
    parser.add_argument("--timeout-seconds", type=float, default=settings.horse_api_timeout_seconds)
    parser.add_argument("--retry-attempts", type=int, default=1)
    parser.add_argument("--retry-backoff-seconds", type=float, default=settings.horse_api_retry_backoff_seconds)
    parser.add_argument("--min-request-interval-seconds", type=float, default=settings.horse_api_min_request_interval_seconds)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--no-probe-endpoints", dest="probe_endpoints", action="store_false")
    parser.set_defaults(probe_endpoints=True)
    parser.add_argument("--min-current-rows", type=int)
    parser.add_argument("--min-historical-rows", type=int)
    parser.add_argument("--min-field-coverage", action="append", default=[], help="FIELD=VALUE, for example track=1.0")
    parser.add_argument("--max-error-issues", type=int, default=0)
    parser.add_argument("--max-warning-issues", type=int)
    parser.add_argument("--allow-failure", action="store_true", help="Print failed smoke output but exit zero.")
    parser.add_argument("--output", help="Optional JSON output file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, exit_code = run_smoke(args)
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
