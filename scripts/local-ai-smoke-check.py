from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_ai import LocalAIConfig, redacted_ai_config, review_payload_mapping  # noqa: E402
from settings import get_settings  # noqa: E402


def sample_payload_summary() -> dict[str, Any]:
    return {
        "type": "object",
        "keys": ["course", "race_time", "runners"],
        "listFields": {
            "runners": {
                "rows": 2,
                "sampleKeys": ["horse_name", "jockey_name", "trainer_name", "draw"],
            }
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run a no-secret local OpenAI-compatible AI smoke check.")
    parser.add_argument("--provider", default=settings.ai_provider)
    parser.add_argument("--base-url", default=settings.ai_base_url)
    parser.add_argument("--model", default=settings.ai_model)
    parser.add_argument("--timeout-seconds", type=float, default=settings.ai_timeout_seconds)
    parser.add_argument("--json-schema-required", action="store_true", default=settings.ai_json_schema_required)
    parser.add_argument("--no-json-schema-required", dest="json_schema_required", action="store_false")
    parser.add_argument("--api-key", default=settings.ai_api_key)
    parser.add_argument("--allow-failure", action="store_true", help="Print failed smoke output but exit zero.")
    parser.add_argument("--output", help="Optional JSON output file.")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> LocalAIConfig:
    return LocalAIConfig(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        json_schema_required=args.json_schema_required,
        api_key=args.api_key,
    )


def run_smoke(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = build_config(args)
    payload: dict[str, Any] = {"config": redacted_ai_config(config)}
    try:
        review = review_payload_mapping(sample_payload_summary(), config)
    except Exception as exc:
        payload["status"] = "failed"
        payload["message"] = str(exc)
        return payload, 0 if args.allow_failure else 1

    payload["status"] = "passed" if review.get("status") in {"success", "disabled"} else str(review.get("status"))
    payload["review"] = review
    return payload, 0 if payload["status"] == "passed" or args.allow_failure else 1


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
