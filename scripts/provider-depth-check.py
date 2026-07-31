from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from race_enrichment import course_metadata_payload, fetch_open_meteo_daily_weather


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run provider-depth enrichment checks.")
    parser.add_argument("--track", default="York", help="Known racecourse name used for course metadata lookup.")
    parser.add_argument("--race-date", default="2025-03-18", help="ISO date for the Open-Meteo historical weather check.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--skip-weather", action="store_true", help="Only validate local course metadata.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    course = course_metadata_payload(args.track)
    if not course:
        print(f"No course metadata found for {args.track!r}.", file=sys.stderr)
        return 2

    payload = {"track": args.track, "course": course}
    if not args.skip_weather:
        payload["weather"] = fetch_open_meteo_daily_weather(course["latitude"], course["longitude"], args.race_date, timeout=args.timeout)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
