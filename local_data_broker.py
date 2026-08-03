from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import pandas as pd

from provider_adapters import ProviderResult, RawProviderPayload, extract_records, normalize_provider_records

BROKER_SCHEMA_VERSION = "2026-08-03.1"
SAFE_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class CachedPayloadRecord:
    provider: str
    resource: str
    path: str
    payload_sha256: str
    fetched_at: str
    endpoint: str | None = None
    source_url: str | None = None
    row_count: int = 0
    license_reference: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str, fallback: str = "payload") -> str:
    candidate = SAFE_SLUG_PATTERN.sub("-", value.strip()).strip("-._").lower()
    return candidate or fallback


def json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, default=json_default, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def endpoint_row_count(payload: dict | list) -> int:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return 0
    known_lengths = []
    for key in ["results", "racecards", "data", "courses", "runners", "historical", "current"]:
        value = payload.get(key)
        if isinstance(value, list):
            known_lengths.append(len(value))
    if known_lengths:
        return sum(known_lengths)
    list_lengths = [len(value) for value in payload.values() if isinstance(value, list)]
    return sum(list_lengths) if list_lengths else len(payload)


def build_source_url(base_url: str, endpoint: str, params: dict | None = None) -> str | None:
    if not base_url and not endpoint:
        return None
    source_url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}" if base_url else endpoint
    if params:
        source_url = f"{source_url}?{urlencode(params, doseq=True)}"
    return source_url


def cache_file_name(resource: str, fetched_at: str, payload_hash: str) -> str:
    timestamp = fetched_at.replace("+00:00", "Z").replace(":", "").replace("-", "")
    return f"{timestamp}-{slugify(resource)}-{payload_hash[:12]}.json"


def write_raw_payload(
    cache_dir: str | Path,
    provider: str,
    resource: str,
    payload: dict | list,
    *,
    endpoint: str | None = None,
    params: dict | None = None,
    source_url: str | None = None,
    license_reference: str | None = None,
    fetched_at: str | None = None,
) -> CachedPayloadRecord:
    fetched_at = fetched_at or utc_now_iso()
    payload_hash = payload_sha256(payload)
    provider_slug = slugify(provider, "provider")
    resource_slug = slugify(resource)
    target_dir = Path(cache_dir) / provider_slug / resource_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / cache_file_name(resource, fetched_at, payload_hash)
    envelope = {
        "schemaVersion": BROKER_SCHEMA_VERSION,
        "metadata": {
            "provider": provider,
            "resource": resource,
            "endpoint": endpoint,
            "params": params or {},
            "sourceUrl": source_url,
            "licenseReference": license_reference,
            "fetchedAt": fetched_at,
            "payloadSha256": payload_hash,
            "rowCount": endpoint_row_count(payload),
        },
        "payload": payload,
    }
    target_path.write_text(json.dumps(envelope, default=json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cached_payload_record(target_path, envelope)


def cached_payload_record(path: Path, envelope: dict[str, Any]) -> CachedPayloadRecord:
    metadata = envelope.get("metadata", {})
    return CachedPayloadRecord(
        provider=str(metadata.get("provider") or "unknown"),
        resource=str(metadata.get("resource") or path.parent.name),
        path=str(path),
        payload_sha256=str(metadata.get("payloadSha256") or ""),
        fetched_at=str(metadata.get("fetchedAt") or ""),
        endpoint=metadata.get("endpoint"),
        source_url=metadata.get("sourceUrl"),
        row_count=int(metadata.get("rowCount") or 0),
        license_reference=metadata.get("licenseReference"),
    )


def load_cached_payload(path: str | Path) -> tuple[dict[str, Any], dict | list]:
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = envelope.get("metadata", {})
    payload = envelope.get("payload")
    if not isinstance(metadata, dict) or not isinstance(payload, (dict, list)):
        raise ValueError(f"Cached broker payload is invalid: {path}")
    expected_hash = metadata.get("payloadSha256")
    actual_hash = payload_sha256(payload)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(f"Cached broker payload hash mismatch: {path}")
    return metadata, payload


def list_cached_payloads(cache_dir: str | Path, provider: str | None = None, limit: int = 50) -> list[CachedPayloadRecord]:
    root = Path(cache_dir)
    if not root.exists():
        return []
    candidates = root.glob("**/*.json")
    records: list[CachedPayloadRecord] = []
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            record = cached_payload_record(path, envelope)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if provider and record.provider != provider:
            continue
        records.append(record)
        if len(records) >= limit:
            break
    return records


def cache_provider_payloads(
    cache_dir: str | Path,
    provider: str,
    payloads: Iterable[RawProviderPayload],
    *,
    base_url: str = "",
    license_reference: str | None = None,
) -> list[CachedPayloadRecord]:
    records = []
    for raw_payload in payloads:
        source_url = build_source_url(base_url, raw_payload.endpoint, raw_payload.params)
        records.append(
            write_raw_payload(
                cache_dir,
                provider,
                raw_payload.resource,
                raw_payload.payload,
                endpoint=raw_payload.endpoint,
                params=raw_payload.params,
                source_url=source_url,
                license_reference=license_reference,
            )
        )
    return records


def raw_payload_cache_status(cache_dir: str | Path, limit: int = 500) -> dict[str, Any]:
    records = list_cached_payloads(cache_dir, limit=limit)
    by_provider: dict[str, int] = {}
    by_resource: dict[str, int] = {}
    latest = None
    for record in records:
        by_provider[record.provider] = by_provider.get(record.provider, 0) + 1
        by_resource[record.resource] = by_resource.get(record.resource, 0) + 1
        if record.fetched_at and (latest is None or record.fetched_at > latest):
            latest = record.fetched_at
    return {
        "cacheDir": str(Path(cache_dir)),
        "rawPayloadCount": len(records),
        "providers": by_provider,
        "resources": by_resource,
        "latestFetchedAt": latest,
    }


def payload_shape(payload: dict | list) -> dict[str, Any]:
    if isinstance(payload, list):
        sample = payload[0] if payload and isinstance(payload[0], dict) else {}
        return {"type": "list", "rows": len(payload), "sampleKeys": sorted(sample.keys())[:50]}
    shape = {"type": "object", "keys": sorted(payload.keys())[:50], "listFields": {}}
    for key, value in payload.items():
        if isinstance(value, list):
            sample = value[0] if value and isinstance(value[0], dict) else {}
            shape["listFields"][key] = {"rows": len(value), "sampleKeys": sorted(sample.keys())[:50]}
    return shape


def manual_json_to_provider_result(payload: dict[str, Any]) -> ProviderResult:
    historical_records = extract_records(payload.get("historical", []), ["historical", "results", "races"])
    current_records = extract_records(payload.get("current", []), ["current", "racecards", "races"])
    if not historical_records and "results" in payload:
        historical_records = extract_records(payload, ["results"])
    if not current_records and "racecards" in payload:
        current_records = extract_records(payload, ["racecards"])

    historical_df, historical_issues = normalize_provider_records(historical_records, current_mode=False)
    current_df, current_issues = normalize_provider_records(current_records, current_mode=True)
    return ProviderResult(
        historical=historical_df,
        current=current_df,
        validation_issues=historical_issues + current_issues,
        raw_payloads=[RawProviderPayload("manual_json", "manual://json", payload)],
    )


def provider_result_summary(result: ProviderResult) -> dict[str, Any]:
    return {
        "historicalRows": int(len(result.historical)),
        "currentRows": int(len(result.current)),
        "validationIssues": len(result.validation_issues),
        "rawPayloads": len(result.raw_payloads),
    }


def frame_to_broker_rows(frame: pd.DataFrame, limit: int = 100) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.head(limit).where(pd.notna(frame), None).to_dict(orient="records")
