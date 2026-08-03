from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from racing_storage import create_or_update_operator_account, read_operator_account_by_token  # noqa: E402
from settings import get_settings  # noqa: E402


def token_state(token: str | None) -> dict[str, Any]:
    cleaned = (token or "").strip()
    return {"present": bool(cleaned), "length": len(cleaned)}


def account_summary(account: dict[str, Any] | None) -> dict[str, Any] | None:
    if account is None:
        return None
    return {
        key: value
        for key, value in account.items()
        if key not in {"tokenSha256"}
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Create or update an operator account and verify token lookup.")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--account-key", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--email")
    parser.add_argument("--roles", default="viewer", help="Comma-separated roles such as viewer,journal,operator.")
    parser.add_argument("--token", help="Bearer token to hash and store. Minimum 16 characters.")
    parser.add_argument("--status", default="active", choices=["active", "disabled"])
    parser.add_argument("--privacy-acknowledged", action="store_true")
    parser.add_argument("--output", help="Optional JSON output file.")
    return parser.parse_args(argv)


def run_smoke(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    account = create_or_update_operator_account(
        args.database_url,
        account_key=args.account_key,
        display_name=args.display_name,
        email=args.email,
        roles=args.roles,
        token=args.token,
        status=args.status,
        privacy_acknowledged=args.privacy_acknowledged,
    )
    matched = read_operator_account_by_token(args.database_url, args.token) if args.token else None
    status = "passed" if not args.token or matched else "failed"
    payload = {
        "status": status,
        "databaseUrl": args.database_url,
        "token": token_state(args.token),
        "account": account_summary(account),
        "lookup": {
            "matched": matched is not None,
            "account": account_summary(matched),
        },
    }
    return payload, 0 if status == "passed" else 1


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
