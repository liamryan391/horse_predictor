from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from racing_storage import normalized_table_counts, read_normalized_race_entries, sync_normalized_from_compatibility  # noqa: E402
from settings import get_settings  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Sync flat race tables into normalized provider entity tables.")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--source", help="Optional source/provider filter, for example sample or ourhub.")
    parser.add_argument("--skip-historical", action="store_true")
    parser.add_argument("--skip-current", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", help="Optional JSON output file.")
    return parser.parse_args(argv)


def run_check(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    sync_counts = sync_normalized_from_compatibility(
        args.database_url,
        source=args.source,
        include_historical=not args.skip_historical,
        include_current=not args.skip_current,
    )
    entries = read_normalized_race_entries(args.database_url, provider=args.source, limit=args.limit)
    payload = {
        "status": "passed" if sync_counts.get("skipped", 0) == 0 else "warning",
        "databaseUrl": args.database_url,
        "source": args.source,
        "sync": sync_counts,
        "tables": normalized_table_counts(args.database_url),
        "sampleEntries": entries,
    }
    return payload, 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, exit_code = run_check(args)
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
