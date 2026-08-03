from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
from math import isfinite
from threading import Lock
from typing import Any

import pandas as pd


NUMERIC_DRIFT_FIELDS = [
    "odds",
    "implied_probability",
    "field_size",
    "odds_rank",
    "speed_rating",
    "class_rating",
    "relative_speed_rating",
    "relative_class_rating",
    "draw",
    "horse_age",
    "horse_weight",
    "days_since_last_run",
]

CATEGORICAL_DRIFT_FIELDS = [
    "track",
    "surface",
    "country",
    "distance_bucket",
    "going_category",
    "race_type",
    "weather",
]

PREDICTION_DRIFT_FIELDS = ["win_probability", "model_odds", "value_edge"]
API_RECENT_LIMIT = 200

_api_lock = Lock()
_api_metrics: dict[str, Any] = {
    "totalRequests": 0,
    "errorRequests": 0,
    "slowRequests": 0,
    "totalElapsedMs": 0.0,
    "statusCounts": Counter(),
    "pathCounts": Counter(),
    "recent": deque(maxlen=API_RECENT_LIMIT),
    "lastErrorAt": None,
    "lastSlowAt": None,
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def reset_api_metrics() -> None:
    with _api_lock:
        _api_metrics["totalRequests"] = 0
        _api_metrics["errorRequests"] = 0
        _api_metrics["slowRequests"] = 0
        _api_metrics["totalElapsedMs"] = 0.0
        _api_metrics["statusCounts"] = Counter()
        _api_metrics["pathCounts"] = Counter()
        _api_metrics["recent"] = deque(maxlen=API_RECENT_LIMIT)
        _api_metrics["lastErrorAt"] = None
        _api_metrics["lastSlowAt"] = None


def record_api_request(
    method: str,
    path: str,
    status_code: int,
    elapsed_ms: float,
    slow_request_ms: float = 1000.0,
) -> dict[str, Any]:
    is_error = status_code >= 500
    is_slow = elapsed_ms >= slow_request_ms
    recorded_at = utc_timestamp()
    with _api_lock:
        _api_metrics["totalRequests"] += 1
        _api_metrics["totalElapsedMs"] += elapsed_ms
        _api_metrics["statusCounts"][status_code] += 1
        _api_metrics["pathCounts"][path] += 1
        if is_error:
            _api_metrics["errorRequests"] += 1
            _api_metrics["lastErrorAt"] = recorded_at
        if is_slow:
            _api_metrics["slowRequests"] += 1
            _api_metrics["lastSlowAt"] = recorded_at
        _api_metrics["recent"].append(
            {
                "method": method,
                "path": path,
                "statusCode": status_code,
                "elapsedMs": round(elapsed_ms, 2),
                "recordedAt": recorded_at,
            }
        )
    return {"isError": is_error, "isSlow": is_slow}


def api_metrics_snapshot() -> dict[str, Any]:
    with _api_lock:
        total = int(_api_metrics["totalRequests"])
        errors = int(_api_metrics["errorRequests"])
        slow = int(_api_metrics["slowRequests"])
        average = float(_api_metrics["totalElapsedMs"]) / total if total else 0.0
        status_counts = dict(sorted(_api_metrics["statusCounts"].items()))
        top_paths = [
            {"path": path, "requests": count}
            for path, count in _api_metrics["pathCounts"].most_common(10)
        ]
        recent = list(_api_metrics["recent"])
        return {
            "totalRequests": total,
            "errorRequests": errors,
            "slowRequests": slow,
            "errorRate": errors / total if total else 0.0,
            "averageLatencyMs": round(average, 2),
            "statusCounts": {str(key): value for key, value in status_counts.items()},
            "topPaths": top_paths,
            "recent": recent,
            "lastErrorAt": _api_metrics["lastErrorAt"],
            "lastSlowAt": _api_metrics["lastSlowAt"],
        }


def status_for_score(score: float | None, warning_threshold: float, critical_threshold: float) -> str:
    if score is None:
        return "blocked"
    if score >= critical_threshold:
        return "critical"
    if score >= warning_threshold:
        return "warning"
    return "ok"


def worst_status(statuses: list[str]) -> str:
    rank = {"ok": 0, "warning": 1, "blocked": 2, "critical": 3}
    if not statuses:
        return "blocked"
    return max(statuses, key=lambda status: rank.get(status, 0))


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def clean_number_series(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series([pd.NA] * len(frame), dtype="Float64")
    series = pd.to_numeric(frame[field], errors="coerce")
    return series.where(series.map(lambda value: pd.isna(value) or isfinite(float(value))))


def missing_rate(series: pd.Series, total_rows: int) -> float:
    if total_rows == 0:
        return 1.0
    if len(series) == 0:
        return 1.0
    return float(series.isna().sum() / total_rows)


def numeric_drift_row(
    field: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    warning_threshold: float,
    critical_threshold: float,
    kind: str = "numeric",
) -> dict[str, Any]:
    reference_series = clean_number_series(reference, field)
    current_series = clean_number_series(current, field)
    reference_clean = reference_series.dropna()
    current_clean = current_series.dropna()
    reference_rows = len(reference)
    current_rows = len(current)
    reference_mean = finite_float(reference_clean.mean()) if not reference_clean.empty else None
    current_mean = finite_float(current_clean.mean()) if not current_clean.empty else None
    reference_median = finite_float(reference_clean.median()) if not reference_clean.empty else None
    current_median = finite_float(current_clean.median()) if not current_clean.empty else None
    reference_missing_rate = missing_rate(reference_series, reference_rows)
    current_missing_rate = missing_rate(current_series, current_rows)
    missing_delta = abs(current_missing_rate - reference_missing_rate)

    if reference_mean is None or current_mean is None:
        score = missing_delta if reference_rows and current_rows else None
    else:
        std_value = finite_float(reference_clean.std()) or 0.0
        scale = max(abs(reference_mean), abs(std_value), 1.0)
        mean_delta = abs(current_mean - reference_mean) / scale
        median_delta = 0.0
        if reference_median is not None and current_median is not None:
            median_delta = abs(current_median - reference_median) / scale
        score = max(mean_delta, median_delta, missing_delta)

    status = status_for_score(score, warning_threshold, critical_threshold)
    return {
        "field": field,
        "kind": kind,
        "status": status,
        "score": score,
        "referenceCount": int(reference_clean.count()),
        "currentCount": int(current_clean.count()),
        "referenceMean": reference_mean,
        "currentMean": current_mean,
        "referenceMissingRate": reference_missing_rate,
        "currentMissingRate": current_missing_rate,
        "detail": f"Missing-rate delta {missing_delta:.1%}.",
    }


def category_distribution(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns or frame.empty:
        return pd.Series(dtype=float)
    values = frame[field].fillna("__missing__").astype(str).str.strip()
    values = values.where(values != "", "__missing__")
    return values.value_counts(normalize=True, dropna=False)


def categorical_drift_row(
    field: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    warning_threshold: float,
    critical_threshold: float,
) -> dict[str, Any]:
    reference_dist = category_distribution(reference, field)
    current_dist = category_distribution(current, field)
    categories = set(reference_dist.index.tolist()) | set(current_dist.index.tolist())
    if not categories:
        score = None
        max_share_delta = None
    else:
        deltas = [abs(float(current_dist.get(category, 0.0)) - float(reference_dist.get(category, 0.0))) for category in categories]
        max_share_delta = max(deltas) if deltas else 0.0
        new_share = sum(float(current_dist.get(category, 0.0)) for category in categories if category not in reference_dist.index)
        score = max(max_share_delta, new_share)
    status = status_for_score(score, warning_threshold, critical_threshold)
    new_categories = sorted(str(category) for category in categories if category not in reference_dist.index and category != "__missing__")
    top_reference = str(reference_dist.idxmax()) if not reference_dist.empty else None
    top_current = str(current_dist.idxmax()) if not current_dist.empty else None
    return {
        "field": field,
        "kind": "categorical",
        "status": status,
        "score": score,
        "referenceCount": int(len(reference)),
        "currentCount": int(len(current)),
        "referenceShare": float(reference_dist.get(top_reference, 0.0)) if top_reference else None,
        "currentShare": float(current_dist.get(top_current, 0.0)) if top_current else None,
        "topReferenceCategory": None if top_reference == "__missing__" else top_reference,
        "topCurrentCategory": None if top_current == "__missing__" else top_current,
        "maxShareDelta": max_share_delta,
        "newCategories": new_categories[:8],
        "detail": f"{len(new_categories)} new categories in current data.",
    }


def build_drift_report(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    reference_predictions: pd.DataFrame | None = None,
    current_predictions: pd.DataFrame | None = None,
    warning_threshold: float = 0.35,
    critical_threshold: float = 0.75,
) -> dict[str, Any]:
    feature_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for field in NUMERIC_DRIFT_FIELDS:
        if field in reference_features.columns or field in current_features.columns:
            feature_rows.append(numeric_drift_row(field, reference_features, current_features, warning_threshold, critical_threshold))

    for field in CATEGORICAL_DRIFT_FIELDS:
        if field in reference_features.columns or field in current_features.columns:
            feature_rows.append(categorical_drift_row(field, reference_features, current_features, warning_threshold, critical_threshold))

    if reference_predictions is not None and current_predictions is not None:
        for field in PREDICTION_DRIFT_FIELDS:
            if field in reference_predictions.columns or field in current_predictions.columns:
                prediction_rows.append(
                    numeric_drift_row(
                        field,
                        reference_predictions,
                        current_predictions,
                        warning_threshold,
                        critical_threshold,
                        kind="prediction",
                    )
                )

    statuses = [row["status"] for row in feature_rows + prediction_rows]
    return {
        "status": worst_status(statuses),
        "generatedAt": utc_timestamp(),
        "referenceRows": int(len(reference_features)),
        "currentRows": int(len(current_features)),
        "thresholds": {"warning": warning_threshold, "critical": critical_threshold},
        "featureDrift": feature_rows,
        "predictionDrift": prediction_rows,
    }


def drift_alerts(drift_report: dict[str, Any]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for row in drift_report.get("featureDrift", []) + drift_report.get("predictionDrift", []):
        if row.get("status") not in {"warning", "critical"}:
            continue
        field = row.get("field")
        kind = row.get("kind")
        severity = "critical" if row.get("status") == "critical" else "warning"
        alerts.append(
            {
                "severity": severity,
                "category": "drift",
                "code": f"{kind}_{field}_drift",
                "message": f"{field} {kind} drift is {row.get('status')}.",
                "value": row.get("score"),
                "threshold": (drift_report.get("thresholds") or {}).get(severity),
            }
        )
    return alerts
