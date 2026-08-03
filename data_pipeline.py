from __future__ import annotations

import argparse
import os
import platform
import time
from datetime import date, timedelta

from local_data_broker import cache_provider_payloads
from observability import configure_logging
from provider_adapters import APIConfig, FetchContext, get_provider_adapter, summarize_validation_issues
from racing_storage import TABLES, record_ingestion_run, release_job_lock, seed_database_from_samples, sync_normalized_from_compatibility, try_acquire_job_lock, write_races
from settings import get_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Fetch horse racing data, store it in SQL, and optionally export CSV snapshots."
    )
    parser.add_argument(
        "--provider",
        choices=["sample", "generic", "theracingapi", "ourhub"],
        default=settings.horse_api_provider,
        help="sample keeps the app usable without credentials; generic expects JSON matching the app schema.",
    )
    parser.add_argument("--base-url", default=settings.horse_api_base_url)
    parser.add_argument("--api-key", default=settings.horse_api_key)
    parser.add_argument("--username", default=settings.racing_api_username)
    parser.add_argument("--password", default=settings.racing_api_password)
    parser.add_argument(
        "--database-url",
        default=settings.database_url,
        help="SQLAlchemy database URL. Use mysql+pymysql://user:pass@host:3306/horse_predictor for MySQL.",
    )
    parser.add_argument("--historical-out", default=str(settings.sample_historical_csv))
    parser.add_argument("--current-out", default=str(settings.sample_current_csv))
    parser.add_argument("--days-ahead", type=int, default=7)
    parser.add_argument("--history-start", default=(date.today() - timedelta(days=365)).isoformat())
    parser.add_argument("--history-end", default=date.today().isoformat())
    parser.add_argument("--timeout-seconds", type=float, default=settings.horse_api_timeout_seconds)
    parser.add_argument("--retry-attempts", type=int, default=settings.horse_api_retry_attempts)
    parser.add_argument("--retry-backoff-seconds", type=float, default=settings.horse_api_retry_backoff_seconds)
    parser.add_argument("--min-request-interval-seconds", type=float, default=settings.horse_api_min_request_interval_seconds)
    parser.add_argument("--max-pages", type=int, default=settings.horse_api_max_pages)
    parser.add_argument("--no-csv", action="store_true", help="Only update SQL; do not write CSV snapshots.")
    parser.add_argument("--cache-raw-payloads", action="store_true", help="Cache raw provider payloads for local broker replay.")
    parser.add_argument("--broker-cache-dir", default=str(settings.broker_raw_cache_dir))
    parser.add_argument("--skip-normalized-sync", action="store_true", help="Skip syncing compatibility rows into normalized provider entity tables.")
    parser.add_argument("--repeat-hourly", action="store_true", help="Keep the ingestion worker running hourly.")
    parser.add_argument("--disable-lock", action="store_true", help="Run without acquiring the ingestion worker lock.")
    parser.add_argument("--lock-name", default="ingestion-worker")
    parser.add_argument("--lock-ttl-seconds", type=int, default=55 * 60)
    return parser.parse_args(argv)


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
        normalized = {} if args.skip_normalized_sync else sync_normalized_from_compatibility(args.database_url, source="sample")
        return {
            "historical": counts["historical"],
            "current": counts["current"],
            "source": "sample",
            "normalized_entries": normalized.get("raceEntries", 0),
        }

    config = APIConfig(
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        username=args.username,
        password=args.password,
        timeout_seconds=args.timeout_seconds,
        retry_attempts=args.retry_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        min_request_interval_seconds=args.min_request_interval_seconds,
        max_pages=args.max_pages,
    )

    context = FetchContext(args.days_ahead, args.history_start, args.history_end)
    try:
        adapter = get_provider_adapter(config)
        provider_result = adapter.fetch(context)
    except Exception as exc:
        record_ingestion_run(args.database_url, args.provider, "provider_fetch", "failure", 0, str(exc))
        raise

    historical_df = provider_result.historical
    current_df = provider_result.current
    source = args.provider
    validation_message = summarize_validation_issues(provider_result.validation_issues)
    cached_payloads = []
    if args.cache_raw_payloads:
        settings = get_settings()
        cached_payloads = cache_provider_payloads(
            args.broker_cache_dir,
            args.provider,
            provider_result.raw_payloads,
            base_url=adapter.config.base_url,
            license_reference=settings.data_license_reference or None,
        )

    export_snapshots(historical_df, current_df, args)
    historical_count = 0
    if not historical_df.empty:
        historical_count = write_races(
            args.database_url,
            TABLES["historical"],
            historical_df,
            source=source,
            message=validation_message,
        )
    current_count = 0
    if not current_df.empty:
        current_count = write_races(
            args.database_url,
            TABLES["current"],
            current_df,
            source=source,
            message=validation_message,
        )
    normalized = {} if args.skip_normalized_sync else sync_normalized_from_compatibility(args.database_url, source=source)
    return {
        "historical": historical_count,
        "current": current_count,
        "source": source,
        "validation_issues": len(provider_result.validation_issues),
        "cached_payloads": len(cached_payloads),
        "normalized_entries": normalized.get("raceEntries", 0),
    }


def run_once_with_lock(args: argparse.Namespace) -> dict:
    if args.disable_lock:
        return run_once(args)

    owner = f"{platform.node() or 'host'}:{os.getpid()}"
    acquired = try_acquire_job_lock(
        args.database_url,
        args.lock_name,
        owner,
        args.lock_ttl_seconds,
        message=f"provider={args.provider}",
    )
    if not acquired:
        return {"historical": 0, "current": 0, "source": args.provider, "validation_issues": 0, "skipped": True}

    try:
        return run_once(args)
    finally:
        release_job_lock(args.database_url, args.lock_name, owner)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.app_env, settings.log_format)
    while True:
        result = run_once_with_lock(args)
        if result.get("skipped"):
            print(f"Skipped {args.provider} ingestion because lock {args.lock_name} is already active.")
        else:
            print(
                f"Updated {args.database_url}: {result['historical']} historical rows, "
                f"{result['current']} current rows from {result['source']} "
                f"({result.get('validation_issues', 0)} validation issues, "
                f"{result.get('cached_payloads', 0)} cached payloads, "
                f"{result.get('normalized_entries', 0)} normalized entries)."
            )
        if not args.repeat_hourly:
            break
        time.sleep(60 * 60)


if __name__ == "__main__":
    main()
