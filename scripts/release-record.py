from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any
import urllib.error
import urllib.request


PUBLIC_ENDPOINTS = [
    "/api/v1/health",
    "/api/v1/ready",
    "/api/v1/summary",
    "/api/v1/monitoring",
    "/api/v1/model",
    "/api/v1/model/registry",
    "/api/v1/prediction-runs",
    "/api/v1/ingestion-status",
]


def request_json(
    base_url: str,
    path: str,
    timeout: float,
    token: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if actor:
        headers["X-Admin-Actor"] = actor
        headers["X-Journal-Actor"] = actor
    request = urllib.request.Request(f"{base_url}{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{path} could not be reached: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} did not return JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} returned an unexpected JSON shape.")
    return payload


def run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_value(args: list[str]) -> str | None:
    return run_text(["git", *args])


def alembic_value(args: list[str]) -> str | None:
    return run_text([sys.executable, "-m", "alembic", *args])


def first_matching(items: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any] | None:
    for item in items:
        if item.get(key) == value:
            return item
    return None


def summarize_alerts(monitoring: dict[str, Any]) -> dict[str, Any]:
    alerts = monitoring.get("alerts") or []
    critical = [alert for alert in alerts if alert.get("severity") in {"critical", "blocked"}]
    return {
        "status": monitoring.get("status"),
        "total": len(alerts),
        "criticalOrBlocked": len(critical),
        "codes": [alert.get("code") for alert in alerts],
        "criticalCodes": [alert.get("code") for alert in critical],
    }


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    payloads = {path: request_json(base_url, path, args.timeout) for path in PUBLIC_ENDPOINTS}

    registry_models = payloads["/api/v1/model/registry"].get("models") or []
    prediction_runs = payloads["/api/v1/prediction-runs"].get("runs") or []
    approved_model = first_matching(registry_models, "status", "approved")

    admin_payload: dict[str, Any] | None = None
    if args.admin_token:
        governance = request_json(
            base_url,
            "/api/v1/admin/governance",
            args.timeout,
            token=args.admin_token,
            actor=args.admin_actor,
        )
        audit = request_json(
            base_url,
            "/api/v1/admin/audit-log?limit=10",
            args.timeout,
            token=args.admin_token,
            actor=args.admin_actor,
        )
        admin_payload = {
            "actor": governance.get("session", {}).get("actor"),
            "roles": governance.get("session", {}).get("roles"),
            "auditEventCount": audit.get("page", {}).get("total"),
            "latestAuditAction": (audit.get("events") or [{}])[0].get("action"),
        }

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "apiBaseUrl": base_url,
        "git": {
            "branch": git_value(["branch", "--show-current"]),
            "commitSha": git_value(["rev-parse", "HEAD"]),
            "workingTreeClean": git_value(["status", "--short"]) == "",
        },
        "migration": {
            "heads": alembic_value(["heads"]),
            "current": alembic_value(["current"]),
        },
        "readiness": {
            "status": payloads["/api/v1/ready"].get("status"),
            "databaseReady": payloads["/api/v1/ready"].get("databaseReady"),
            "modelReady": payloads["/api/v1/ready"].get("modelReady"),
            "message": payloads["/api/v1/ready"].get("message"),
        },
        "dataFreshness": payloads["/api/v1/summary"].get("dataFreshness"),
        "monitoring": summarize_alerts(payloads["/api/v1/monitoring"]),
        "model": {
            "modelVersionId": payloads["/api/v1/model"].get("modelVersionId"),
            "servingMode": payloads["/api/v1/model"].get("servingMode"),
            "artifactUri": payloads["/api/v1/model"].get("artifactUri"),
            "featureCount": payloads["/api/v1/model"].get("featureCount"),
            "evaluationStatus": payloads["/api/v1/model"].get("evaluation", {}).get("status"),
        },
        "approvedModel": {
            "id": approved_model.get("id") if approved_model else None,
            "artifactReady": approved_model.get("artifactReady") if approved_model else False,
            "artifactSha256": approved_model.get("artifactSha256") if approved_model else None,
            "featureSchemaHash": approved_model.get("featureSchemaHash") if approved_model else None,
        },
        "latestPredictionRun": prediction_runs[0] if prediction_runs else None,
        "ingestion": payloads["/api/v1/ingestion-status"].get("ingestion"),
        "adminGovernance": admin_payload,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Horse Predictor release evidence record.")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--admin-token", default=os.getenv("API_AUTH_TOKEN", ""))
    parser.add_argument("--admin-actor", default=os.getenv("ADMIN_ACTOR", "release-record"))
    parser.add_argument("--output", help="Optional path to write the release record JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = build_record(args)
    rendered = json.dumps(record, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Release record written to {output_path}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
