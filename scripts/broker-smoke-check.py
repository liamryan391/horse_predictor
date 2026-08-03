from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_data_broker import (  # noqa: E402
    manual_json_to_provider_result,
    payload_shape,
    provider_result_summary,
    raw_payload_cache_status,
    write_raw_payload,
)
from settings import get_settings  # noqa: E402


def sample_manual_payload() -> dict[str, Any]:
    today = date.today().isoformat()
    return {
        "historical": [
            {
                "race_date": today,
                "track": "York",
                "distance": 1760,
                "surface": "Good",
                "horse": "Broker Winner",
                "jockey": "A. Local",
                "owner": "Local Owner",
                "trainer": "T. Broker",
                "odds": 4.2,
                "finishing_position": 1,
            }
        ],
        "current": [
            {
                "race_date": today,
                "track": "York",
                "distance": 1760,
                "surface": "Good",
                "horse": "Broker Runner",
                "jockey": "A. Local",
                "owner": "Local Owner",
                "trainer": "T. Broker",
                "odds": 5.5,
            }
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run a no-network local data broker smoke check.")
    parser.add_argument("--cache-dir", default=str(settings.broker_raw_cache_dir))
    parser.add_argument("--no-write", action="store_true", help="Normalize and summarize without writing a cache envelope.")
    parser.add_argument("--output", help="Optional JSON output file.")
    return parser.parse_args(argv)


def run_smoke(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    payload = sample_manual_payload()
    result = manual_json_to_provider_result(payload)
    cached = None
    if not args.no_write:
        cached = write_raw_payload(
            args.cache_dir,
            "manual",
            "broker_smoke",
            payload,
            source_url="manual://broker-smoke",
            license_reference="local smoke fixture",
        )
    response = {
        "status": "passed" if not result.validation_issues else "warning",
        "providerResult": provider_result_summary(result),
        "payloadShape": payload_shape(payload),
        "cachedPayload": None if cached is None else cached.__dict__,
        "cache": raw_payload_cache_status(args.cache_dir),
    }
    return response, 0


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
