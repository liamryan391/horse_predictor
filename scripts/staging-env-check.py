from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

from sqlalchemy.engine import make_url


SENSITIVE_MARKERS = ("TOKEN", "PASSWORD", "KEY", "SECRET", "DATABASE_URL")
DEPLOYED_ENVS = {"staging", "production"}
REQUIRED_DEPLOYED_KEYS = (
    "APP_ENV",
    "DATABASE_URL",
    "BACKEND_CORS_ORIGINS",
    "ALLOWED_HOSTS",
    "ACCOUNT_AUTH_ENABLED",
    "JOURNAL_ACCOUNT_KEY",
    "VITE_API_BASE_URL",
    "MODEL_ARTIFACT_DIR",
    "REQUIRE_APPROVED_MODEL_ARTIFACT",
)
PLACEHOLDER_MARKERS = ("replace", "changeme", "change-me", "your-", "example.com")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        raise ValueError(f"Environment file does not exist: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def bool_value(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def csv_values(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def has_placeholder(value: str | None) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in PLACEHOLDER_MARKERS)


def is_local_host(host: str | None) -> bool:
    return (host or "").lower() in {"", "localhost", "127.0.0.1", "::1"} or str(host or "").lower().endswith(".local")


def redact_value(key: str, value: str) -> str:
    if "DATABASE_URL" in key:
        try:
            return make_url(value).render_as_string(hide_password=True)
        except Exception:
            return "<invalid-url>"
    if any(marker in key for marker in SENSITIVE_MARKERS):
        return f"<set:{len(value)} chars>" if value else ""
    return value


def provider_credential_errors(values: dict[str, str]) -> list[str]:
    provider = values.get("HORSE_API_PROVIDER", "sample").strip().lower()
    errors: list[str] = []
    if provider == "ourhub" and not values.get("HORSE_API_KEY"):
        errors.append("HORSE_API_KEY is required when HORSE_API_PROVIDER=ourhub.")
    if provider == "theracingapi":
        if not values.get("RACING_API_USERNAME"):
            errors.append("RACING_API_USERNAME is required when HORSE_API_PROVIDER=theracingapi.")
        if not values.get("RACING_API_PASSWORD"):
            errors.append("RACING_API_PASSWORD is required when HORSE_API_PROVIDER=theracingapi.")
    if provider == "generic" and not values.get("HORSE_API_BASE_URL"):
        errors.append("HORSE_API_BASE_URL is required when HORSE_API_PROVIDER=generic.")
    return errors


def validate(values: dict[str, str], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    app_env = values.get("APP_ENV", "").strip().lower()

    missing = [key for key in REQUIRED_DEPLOYED_KEYS if not values.get(key)]
    errors.extend(f"{key} is required for staging/production." for key in missing)

    if app_env not in DEPLOYED_ENVS:
        warnings.append(f"APP_ENV is {app_env!r}; this checker is intended for staging or production.")

    database_url = values.get("DATABASE_URL", "")
    if database_url:
        try:
            parsed = make_url(database_url)
            if parsed.get_backend_name() == "sqlite":
                errors.append("DATABASE_URL must use managed SQL, not SQLite.")
            if is_local_host(parsed.host):
                errors.append("DATABASE_URL host must not be local for staging/production.")
        except Exception as exc:
            errors.append(f"DATABASE_URL is invalid: {exc}")

    origins = csv_values(values.get("BACKEND_CORS_ORIGINS"))
    if "*" in origins:
        errors.append("BACKEND_CORS_ORIGINS must not include '*'.")
    insecure_origins = [origin for origin in origins if not origin.startswith("https://")]
    if insecure_origins:
        errors.append(f"BACKEND_CORS_ORIGINS must use HTTPS: {', '.join(insecure_origins)}.")

    allowed_hosts = csv_values(values.get("ALLOWED_HOSTS"))
    if "*" in allowed_hosts:
        errors.append("ALLOWED_HOSTS must not include '*'.")

    vite_url = values.get("VITE_API_BASE_URL", "")
    if vite_url and not vite_url.startswith("https://"):
        errors.append("VITE_API_BASE_URL must use HTTPS in staging/production.")

    account_auth_enabled = bool_value(values.get("ACCOUNT_AUTH_ENABLED"))
    if not account_auth_enabled:
        if not values.get("API_AUTH_TOKEN"):
            errors.append("API_AUTH_TOKEN is required when ACCOUNT_AUTH_ENABLED=false.")
        if not values.get("JOURNAL_AUTH_TOKEN"):
            errors.append("JOURNAL_AUTH_TOKEN is required when ACCOUNT_AUTH_ENABLED=false.")

    for key in ["API_AUTH_TOKEN", "JOURNAL_AUTH_TOKEN"]:
        token = values.get(key, "")
        if token and len(token) < 32:
            errors.append(f"{key} must be at least 32 characters when configured.")

    if args.require_release_token and not values.get("ACCOUNT_AUTH_TOKEN"):
        errors.append("ACCOUNT_AUTH_TOKEN is required for release helper admin checks.")

    if args.require_policy_links:
        for key in ["RESPONSIBLE_GAMBLING_URL", "PRIVACY_POLICY_URL", "TERMS_OF_USE_URL"]:
            if not values.get(key):
                errors.append(f"{key} is required when --require-policy-links is set.")

    if args.require_provider_credentials:
        errors.extend(provider_credential_errors(values))

    if values.get("REQUIRE_APPROVED_MODEL_ARTIFACT") and not bool_value(values.get("REQUIRE_APPROVED_MODEL_ARTIFACT")):
        warnings.append("REQUIRE_APPROVED_MODEL_ARTIFACT is not true; staging acceptance should turn it on.")

    if not args.allow_placeholders:
        for key, value in values.items():
            if value and has_placeholder(value):
                errors.append(f"{key} still looks like a placeholder.")

    return errors, warnings


def build_report(values: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    errors, warnings = validate(values, args)
    interesting_keys = [
        "APP_ENV",
        "DATABASE_URL",
        "BACKEND_CORS_ORIGINS",
        "ALLOWED_HOSTS",
        "ACCOUNT_AUTH_ENABLED",
        "ACCOUNT_AUTH_TOKEN",
        "HORSE_API_PROVIDER",
        "VITE_API_BASE_URL",
        "MODEL_ARTIFACT_DIR",
        "REQUIRE_APPROVED_MODEL_ARTIFACT",
    ]
    return {
        "status": "failed" if errors else "passed",
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "envFile": str(args.env_file),
        "errors": errors,
        "warnings": warnings,
        "summary": {key: redact_value(key, values.get(key, "")) for key in interesting_keys if key in values},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate staging/production environment configuration before deployment.")
    parser.add_argument("--env-file", type=Path, default=Path(".env.staging.example"))
    parser.add_argument("--use-current-env", action="store_true", help="Overlay current process environment on top of --env-file.")
    parser.add_argument("--allow-placeholders", action="store_true", help="Allow example placeholder values.")
    parser.add_argument("--require-release-token", action="store_true", help="Require ACCOUNT_AUTH_TOKEN for release helper scripts.")
    parser.add_argument("--require-policy-links", action="store_true")
    parser.add_argument("--require-provider-credentials", action="store_true")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        values = parse_env_file(args.env_file)
    except ValueError as exc:
        print(json.dumps({"status": "failed", "errors": [str(exc)]}, indent=2, sort_keys=True))
        return 1
    if args.use_current_env:
        values.update({key: value for key, value in os.environ.items() if value})

    report = build_report(values, args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
