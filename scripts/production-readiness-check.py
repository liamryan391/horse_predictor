from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


REQUIRED_ENDPOINTS = [
    "/api/v1/health",
    "/api/v1/ready",
    "/api/v1/summary",
    "/api/v1/data-quality",
    "/api/v1/monitoring",
    "/api/v1/safeguards",
    "/api/v1/model/registry",
    "/api/v1/model/evaluation",
    "/api/v1/prediction-runs",
    "/api/v1/ingestion-status",
]

REQUIRED_POLICY_LINKS = ["responsibleGambling", "privacyPolicy", "termsOfUse"]
REQUIRED_ENRICHMENT_FIELDS = ["country", "distance_bucket", "going_category", "race_type"]
REQUIRED_SECURITY_HEADERS = [
    "cache-control",
    "cross-origin-opener-policy",
    "referrer-policy",
    "x-content-type-options",
    "x-frame-options",
    "x-request-id",
]


def request_json(base_url: str, path: str, timeout: float) -> tuple[dict[str, Any], dict[str, str]]:
    url = f"{base_url}{path}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            headers = {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{path} could not be reached: {exc.reason}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} did not return JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} returned an unexpected JSON shape.")
    return payload, headers


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def check_readiness(payloads: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    ready = payloads["/api/v1/ready"]
    summary = payloads["/api/v1/summary"]
    safeguards = payloads["/api/v1/safeguards"]
    data_quality = payloads["/api/v1/data-quality"]
    monitoring = payloads["/api/v1/monitoring"]
    registry = payloads["/api/v1/model/registry"]
    prediction_runs = payloads["/api/v1/prediction-runs"]

    require(ready.get("status") == "ok" or args.allow_degraded, "/api/v1/ready must return status=ok.", failures)
    require(bool(ready.get("databaseReady")) or args.allow_degraded, "Database readiness must be true.", failures)
    require(bool(ready.get("modelReady")) or args.allow_degraded, "Model readiness must be true.", failures)

    freshness = summary.get("dataFreshness") or {}
    require(
        freshness.get("status") == "fresh" or args.allow_stale,
        f"Data freshness must be fresh; got {freshness.get('status')!r}.",
        failures,
    )

    notice = str(safeguards.get("responsibleUseNotice") or "").lower()
    require("not betting advice" in notice, "Safeguards must state that predictions are not betting advice.", failures)
    require(bool(safeguards.get("limitations")), "Safeguards must list model limitations.", failures)
    require(bool(safeguards.get("dataLicensingNotice")), "Safeguards must include a data licensing notice.", failures)
    require(bool(safeguards.get("privacyNotice")), "Safeguards must include a privacy notice.", failures)
    require(bool(safeguards.get("termsNotice")), "Safeguards must include a terms notice.", failures)

    if args.require_policy_links:
        links = safeguards.get("links") or {}
        missing = [link for link in REQUIRED_POLICY_LINKS if not links.get(link)]
        require(not missing, f"Policy links must be configured: {', '.join(missing)}.", failures)

    if args.require_approved_model:
        models = registry.get("models") or []
        approved = [model for model in models if model.get("status") == "approved"]
        require(bool(approved), "At least one approved model version must exist in /api/v1/model/registry.", failures)

    if args.require_approved_artifact:
        models = registry.get("models") or []
        approved_artifacts = [
            model
            for model in models
            if model.get("status") == "approved"
            and model.get("artifactReady")
            and model.get("artifactUri")
            and model.get("artifactSha256")
            and model.get("featureSchemaHash")
        ]
        require(bool(approved_artifacts), "At least one approved model artifact must exist in /api/v1/model/registry.", failures)

    if args.require_prediction_run:
        runs = prediction_runs.get("runs") or []
        require(bool(runs), "At least one persisted prediction run must exist in /api/v1/prediction-runs.", failures)

    if args.require_enriched_data:
        quality_tables = data_quality.get("tables") or []
        require(bool(quality_tables), "/api/v1/data-quality must include table coverage.", failures)
        for table in quality_tables:
            coverage = {row.get("field"): row.get("coverage") for row in table.get("coverage") or []}
            low_fields = [
                field
                for field in REQUIRED_ENRICHMENT_FIELDS
                if not isinstance(coverage.get(field), (float, int)) or coverage[field] < args.min_enrichment_coverage
            ]
            require(
                not low_fields,
                f"{table.get('tableName')} enrichment coverage below {args.min_enrichment_coverage:.0%}: {', '.join(low_fields)}.",
                failures,
            )

    if args.require_monitoring:
        require(
            monitoring.get("status") in {"ok", "warning", "blocked", "critical"},
            "/api/v1/monitoring must return a known status.",
            failures,
        )
        require(bool(monitoring.get("metrics")), "/api/v1/monitoring must include operator metrics.", failures)
        require(isinstance(monitoring.get("apiMetrics"), dict), "/api/v1/monitoring must include API metrics.", failures)
        drift = monitoring.get("drift") or {}
        require(bool(drift.get("featureDrift")), "/api/v1/monitoring must include feature drift checks.", failures)
        require(bool(drift.get("predictionDrift")), "/api/v1/monitoring must include prediction drift checks.", failures)

    if args.require_no_critical_alerts:
        critical_alerts = [
            alert
            for alert in monitoring.get("alerts") or []
            if alert.get("severity") in {"critical", "blocked"}
        ]
        require(
            not critical_alerts,
            f"/api/v1/monitoring has critical alerts: {', '.join(str(alert.get('code')) for alert in critical_alerts)}.",
            failures,
        )

    return failures


def check_security_headers(headers: dict[str, str], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    missing = [header for header in REQUIRED_SECURITY_HEADERS if header not in headers]
    require(not missing, f"Security headers are missing: {', '.join(missing)}.", failures)
    require(headers.get("x-content-type-options") == "nosniff", "X-Content-Type-Options must be nosniff.", failures)
    require(headers.get("x-frame-options") == "DENY", "X-Frame-Options must be DENY.", failures)
    if args.require_hsts:
        require("strict-transport-security" in headers, "Strict-Transport-Security must be present.", failures)
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production-readiness smoke checks against the Horse Predictor API.")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--allow-stale", action="store_true", help="Do not fail when dataFreshness.status is not fresh.")
    parser.add_argument("--allow-degraded", action="store_true", help="Do not fail when /ready is degraded.")
    parser.add_argument("--require-policy-links", action="store_true", help="Require responsible gambling, privacy, and terms URLs.")
    parser.add_argument("--require-approved-model", action="store_true", help="Require an approved model version in the model registry.")
    parser.add_argument("--require-approved-artifact", action="store_true", help="Require an approved model version with artifact integrity metadata.")
    parser.add_argument("--require-prediction-run", action="store_true", help="Require at least one persisted prediction run.")
    parser.add_argument("--require-enriched-data", action="store_true", help="Require core enrichment fields to meet the coverage threshold.")
    parser.add_argument("--min-enrichment-coverage", type=float, default=0.75, help="Minimum required coverage for core enrichment fields.")
    parser.add_argument("--require-monitoring", action="store_true", help="Require the monitoring endpoint to expose metrics and drift checks.")
    parser.add_argument("--require-no-critical-alerts", action="store_true", help="Fail when monitoring reports critical or blocked alerts.")
    parser.add_argument("--require-hsts", action="store_true", help="Require Strict-Transport-Security for HTTPS deployments.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_url = args.base_url.rstrip("/")
    payloads: dict[str, dict[str, Any]] = {}
    response_headers: dict[str, dict[str, str]] = {}

    for path in REQUIRED_ENDPOINTS:
        payloads[path], response_headers[path] = request_json(base_url, path, args.timeout)

    failures = check_readiness(payloads, args)
    failures.extend(check_security_headers(response_headers["/api/v1/health"], args))
    if failures:
        print("Production readiness check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    freshness = payloads["/api/v1/summary"].get("dataFreshness") or {}
    monitoring = payloads["/api/v1/monitoring"]
    ready = payloads["/api/v1/ready"]
    print(f"Production readiness check passed for {base_url}.")
    print(f"Ready: {ready.get('status')}; freshness: {freshness.get('status')}; monitoring: {monitoring.get('status')}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
