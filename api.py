from __future__ import annotations

import contextvars
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from secrets import compare_digest
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status as http_status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.trustedhost import TrustedHostMiddleware

from observability import configure_logging, configure_tracing
from api_contracts import (
    AdminAuditResponse,
    AdminGovernanceResponse,
    AdminSessionResponse,
    BetJournalCreateRequest,
    BetJournalDeleteResponse,
    BetJournalEntryResponse,
    BetJournalListResponse,
    BetJournalUpdateRequest,
    DataQualityResponse,
    EntityProfileResponse,
    ErrorResponse,
    HealthResponse,
    IngestionStatusResponse,
    MeetingsResponse,
    ModelEvaluationResponse,
    ModelRegistryResponse,
    ModelSnapshotResponse,
    ModelStatusResponse,
    MonitoringResponse,
    PageMeta,
    PredictionRunResponse,
    PredictionRunsResponse,
    PredictionsResponse,
    ProductSafeguardsResponse,
    RaceCardResponse,
    RacesResponse,
    ReadinessResponse,
    SeedSampleResponse,
    SummaryResponse,
    TrendsResponse,
)
from monitoring import api_metrics_snapshot, build_drift_report, drift_alerts, record_api_request, utc_timestamp, worst_status
from prediction_model import (
    add_scoring_features,
    artifact_uri_to_path,
    build_feature_table,
    evaluate_model,
    load_model_artifact,
    predict_probabilities,
    save_model_artifact,
    score_current_races,
    summarize_entities,
    train_model,
)
from data_quality import enrichment_quality_issues, field_coverage, race_row_quality_issues
from race_enrichment import ENRICHMENT_COLUMNS, enrich_race_frame
from racing_storage import (
    TABLES,
    approve_model_version,
    delete_bet_journal_entry,
    ingestion_status,
    read_latest_approved_model_version,
    read_admin_audit_events,
    read_bet_journal,
    read_model_version,
    read_model_registry,
    read_prediction_run,
    read_prediction_runs,
    read_races,
    provider_freshness_report,
    record_bet_journal_entry,
    record_admin_audit_event,
    record_model_evaluation_snapshot,
    record_prediction_run,
    seed_database_from_samples,
    table_counts,
    update_model_version_status,
    update_bet_journal_entry,
)
from settings import get_settings

SETTINGS = get_settings()
SETTINGS.validate_runtime()
configure_logging(SETTINGS.app_env, SETTINGS.log_format)
DATABASE_URL = SETTINGS.database_url

SortDirection = Literal["asc", "desc"]
EntityType = Literal["horse", "jockey", "trainer", "owner"]

logger = logging.getLogger("horse_predictor.api")
request_id_context: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
rate_limit_hits: dict[str, deque[float]] = defaultdict(deque)
admin_auth = HTTPBearer(auto_error=False)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
ACTOR_PATTERN = re.compile(r"^[A-Za-z0-9_.:@ -]{1,120}$")


@dataclass(frozen=True)
class AccessContext:
    actor: str
    roles: tuple[str, ...]
    account_key: str

app = FastAPI(
    title="Horse Predictor API",
    version="0.7.0",
    description="Versioned API for race cards, artifact-backed model predictions, ingestion status, model evaluation, and monitoring.",
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(SETTINGS.allowed_hosts))
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.backend_cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Actor", "X-Journal-Actor", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)
TRACING_STATUS = configure_tracing(app, SETTINGS.otel_enabled, SETTINGS.otel_service_name)

router = APIRouter(tags=["v1"])


def request_id() -> str:
    return request_id_context.get()


def normalize_request_id(value: str | None) -> str:
    candidate = (value or "").strip()
    if REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def error_payload(status_code: int, detail: str) -> dict[str, Any]:
    return {"error": {"requestId": request_id(), "statusCode": status_code, "detail": detail}}


def content_length_too_large(request: Request) -> bool:
    content_length = request.headers.get("content-length")
    if not content_length:
        return False
    try:
        return int(content_length) > SETTINGS.max_request_body_bytes
    except ValueError:
        return False


def apply_security_headers(response: Any) -> None:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cache-Control", "no-store")
    if SETTINGS.is_deployed_environment:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")


def client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def is_rate_limited(key: str) -> bool:
    limit = SETTINGS.api_rate_limit_per_minute
    if limit <= 0:
        return False

    now = time.monotonic()
    window_start = now - 60
    hits = rate_limit_hits[key]
    while hits and hits[0] < window_start:
        hits.popleft()
    if len(hits) >= limit:
        return True
    hits.append(now)
    return False


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    active_request_id = normalize_request_id(request.headers.get("x-request-id"))
    token = request_id_context.set(active_request_id)
    start = time.monotonic()
    status_code = 500

    try:
        if content_length_too_large(request):
            response = JSONResponse(
                status_code=413,
                content=error_payload(413, "Request body is too large."),
            )
        elif is_rate_limited(client_key(request)):
            response = JSONResponse(
                status_code=429,
                content=error_payload(429, "API rate limit exceeded. Try again shortly."),
            )
        else:
            response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = active_request_id
        apply_security_headers(response)
        return response
    finally:
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)
        request_flags = record_api_request(
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
            slow_request_ms=SETTINGS.monitoring_slow_request_ms,
        )
        logger.info(
            "api_request",
            extra={
                "request_id": active_request_id,
                "method": request.method,
                "path": request.url.path,
                "elapsed_ms": elapsed_ms,
                "status_code": status_code,
                "slow_request": request_flags["isSlow"],
            },
        )
        request_id_context.reset(token)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content=error_payload(exc.status_code, detail), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content=error_payload(422, str(exc)))


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("api_unhandled_error", extra={"request_id": request_id()})
    detail = "Internal server error." if SETTINGS.is_deployed_environment else str(exc)
    return JSONResponse(status_code=500, content=error_payload(500, detail))


def ensure_seed_data() -> None:
    counts = table_counts(DATABASE_URL)
    if counts["historical"] == 0 or counts["current"] == 0:
        sample_history = SETTINGS.sample_historical_csv
        sample_current = SETTINGS.sample_current_csv
        if sample_history.exists() and sample_current.exists():
            seed_database_from_samples(DATABASE_URL, sample_history, sample_current)


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    return value


def data_freshness(last_refresh: Any) -> dict[str, Any]:
    max_age_hours = SETTINGS.data_freshness_max_age_hours
    if not last_refresh:
        return {"status": "missing", "lastRefresh": None, "ageHours": None, "maxAgeHours": max_age_hours}

    parsed = pd.to_datetime(last_refresh, utc=True, errors="coerce")
    if pd.isna(parsed):
        return {"status": "unknown", "lastRefresh": clean_value(last_refresh), "ageHours": None, "maxAgeHours": max_age_hours}

    age_hours = round((datetime.now(timezone.utc) - parsed.to_pydatetime()).total_seconds() / 3600, 2)
    status = "fresh" if age_hours <= max_age_hours else "stale"
    return {
        "status": status,
        "lastRefresh": clean_value(last_refresh),
        "ageHours": age_hours,
        "maxAgeHours": max_age_hours,
    }


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: clean_value(value) for key, value in row.items()} for row in df.to_dict(orient="records")]


def database_summary() -> dict[str, Any]:
    return SETTINGS.database_summary()


def current_code_commit_sha() -> str | None:
    for env_name in ("SOURCE_COMMIT", "GIT_COMMIT", "VERCEL_GIT_COMMIT_SHA", "RENDER_GIT_COMMIT"):
        value = os.getenv(env_name)
        if value:
            return value.strip()[:80]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()[:80] or None


def current_counts() -> dict[str, int]:
    return table_counts(DATABASE_URL)


def load_registry_model_artifact(model_record: dict[str, Any]) -> Any:
    artifact_uri = model_record.get("artifactUri")
    if not artifact_uri:
        raise ValueError(f"Model version {model_record['id']} has no persisted artifact URI.")

    artifact_path = artifact_uri_to_path(artifact_uri)
    artifact_root = SETTINGS.model_artifact_dir.resolve()
    if not artifact_path.is_relative_to(artifact_root):
        raise ValueError("Model artifact is outside the configured MODEL_ARTIFACT_DIR.")

    return load_model_artifact(
        artifact_uri,
        expected_sha256=model_record.get("artifactSha256"),
        expected_feature_schema_hash=model_record.get("featureSchemaHash"),
    )


def load_model_bundle() -> tuple[pd.DataFrame, pd.DataFrame, Any, pd.DataFrame]:
    ensure_seed_data()
    history_df = read_races(DATABASE_URL, TABLES["historical"])
    current_df = read_races(DATABASE_URL, TABLES["current"])
    history_features = build_feature_table(history_df)
    try:
        model = train_model(history_features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return history_df, current_df, model, history_features


def load_serving_model_bundle() -> tuple[pd.DataFrame, pd.DataFrame, Any, pd.DataFrame, dict[str, Any] | None, str]:
    ensure_seed_data()
    history_df = read_races(DATABASE_URL, TABLES["historical"])
    current_df = read_races(DATABASE_URL, TABLES["current"])
    history_features = build_feature_table(history_df)
    approved_model = read_latest_approved_model_version(DATABASE_URL)

    if approved_model:
        artifact_uri = approved_model.get("artifactUri")
        if artifact_uri:
            try:
                model = load_registry_model_artifact(approved_model)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"Approved model artifact is not loadable: {exc}") from exc
            return history_df, current_df, model, history_features, approved_model, "artifact"

        if SETTINGS.require_approved_model_artifact:
            raise HTTPException(
                status_code=422,
                detail=f"Approved model version {approved_model['id']} has no persisted artifact URI.",
            )

    if SETTINGS.require_approved_model_artifact:
        raise HTTPException(status_code=422, detail="An approved model artifact is required before serving predictions.")

    try:
        model = train_model(history_features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return history_df, current_df, model, history_features, None, "in_memory"


def normalize_date_filter(value: date | None) -> str | None:
    return value.isoformat() if value else None


def text_filter(df: pd.DataFrame, column: str, value: str | None, exact: bool = False) -> pd.DataFrame:
    if not value or column not in df.columns:
        return df
    text = df[column].fillna("").astype(str)
    needle = value.strip()
    if exact:
        return df[text.str.lower() == needle.lower()]
    return df[text.str.contains(needle, case=False, regex=False)]


def filter_race_rows(
    df: pd.DataFrame,
    track: str | None = None,
    race_date: date | None = None,
    horse: str | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    filtered = text_filter(filtered, "track", track, exact=True)
    filtered = text_filter(filtered, "horse", horse)
    target_date = normalize_date_filter(race_date)
    if target_date and "race_date" in filtered.columns:
        filtered = filtered[filtered["race_date"].map(clean_value) == target_date]
    return filtered


def sort_frame(df: pd.DataFrame, sort_by: str, direction: SortDirection) -> pd.DataFrame:
    if sort_by not in df.columns:
        raise HTTPException(status_code=422, detail=f"Unsupported sort field: {sort_by}")
    return df.sort_values(sort_by, ascending=direction == "asc", na_position="last", kind="mergesort")


def page_frame(df: pd.DataFrame, limit: int, offset: int) -> tuple[pd.DataFrame, dict[str, int]]:
    total = len(df)
    page_df = df.iloc[offset : offset + limit].copy()
    return page_df, {"limit": limit, "offset": offset, "returned": len(page_df), "total": total}


def build_page(meta: dict[str, int]) -> PageMeta:
    return PageMeta(**meta)


def request_model_payload(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_unset=True)
    return model.dict(exclude_unset=True)


def build_meetings(current_df: pd.DataFrame) -> pd.DataFrame:
    if current_df.empty:
        return pd.DataFrame(columns=["race_date", "track", "races", "runners", "first_distance", "last_distance"])
    grouped = (
        current_df.groupby(["race_date", "track"], dropna=False)
        .agg(
            races=("distance", "nunique"),
            runners=("horse", "count"),
            first_distance=("distance", "min"),
            last_distance=("distance", "max"),
        )
        .reset_index()
        .sort_values(["race_date", "track"], kind="mergesort")
    )
    return grouped


def build_race_summaries(current_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["race_date", "track", "distance", "surface"]
    for key, race_df in current_df.groupby(group_columns, dropna=False):
        odds = pd.to_numeric(race_df["odds"], errors="coerce")
        favorite = None if odds.dropna().empty else race_df.loc[odds.idxmin(), "horse"]
        rows.append(
            {
                "race_date": key[0],
                "track": key[1],
                "distance": key[2],
                "surface": key[3],
                "runners": int(race_df["horse"].count()),
                "market_favorite": favorite,
                "average_odds": float(odds.mean()) if not odds.dropna().empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["race_date", "track", "distance"], kind="mergesort") if rows else pd.DataFrame(columns=["race_date", "track", "distance", "surface", "runners", "market_favorite", "average_odds"])


def build_scored_predictions(
    race_date: date | None = None,
    track: str | None = None,
    horse: str | None = None,
) -> pd.DataFrame:
    scored_df, _, _ = build_scored_predictions_with_model(race_date=race_date, track=track, horse=horse)
    return scored_df


def build_scored_predictions_with_model(
    race_date: date | None = None,
    track: str | None = None,
    horse: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    _, current_df, model, _, model_record, serving_mode = load_serving_model_bundle()
    scored_df = score_current_races(model, current_df)
    return filter_race_rows(scored_df, track=track, race_date=race_date, horse=horse), model_record, serving_mode


def normalize_actor(value: str | None, fallback: str) -> str:
    candidate = (value or "").strip()
    return candidate if ACTOR_PATTERN.fullmatch(candidate) else fallback


def access_payload(access: AccessContext) -> dict[str, Any]:
    return {
        "requestId": request_id(),
        "actor": access.actor,
        "roles": list(access.roles),
        "environment": SETTINGS.app_env,
        "adminAuthRequired": bool(SETTINGS.api_auth_token) or SETTINGS.is_deployed_environment,
        "journalAuthRequired": bool(SETTINGS.journal_auth_token) or SETTINGS.is_deployed_environment,
        "adminTokenConfigured": bool(SETTINGS.api_auth_token),
        "journalTokenConfigured": bool(SETTINGS.journal_auth_token),
    }


def local_access_context() -> AccessContext:
    return AccessContext(actor="local-dev", roles=("reader", "journal", "admin"), account_key="local")


def access_context_or_local(value: Any, required_role: str = "admin") -> AccessContext:
    if isinstance(value, AccessContext):
        return value
    context = local_access_context()
    if required_role not in context.roles:
        raise HTTPException(status_code=403, detail=f"{required_role} role is required.")
    return context


def access_from_request(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
    required_role: Literal["admin", "journal"],
) -> AccessContext:
    token = credentials.credentials if credentials and credentials.scheme.lower() == "bearer" else None
    actor_header = request.headers.get("x-admin-actor") or request.headers.get("x-journal-actor")

    if SETTINGS.api_auth_token and token and compare_digest(token, SETTINGS.api_auth_token):
        actor = normalize_actor(actor_header, "admin")
        return AccessContext(actor=actor, roles=("reader", "journal", "admin"), account_key="admin")

    if SETTINGS.journal_auth_token and token and compare_digest(token, SETTINGS.journal_auth_token):
        if required_role == "admin":
            raise HTTPException(status_code=403, detail="Administrative role is required.")
        actor = normalize_actor(actor_header, SETTINGS.journal_account_key)
        return AccessContext(actor=actor, roles=("reader", "journal"), account_key=SETTINGS.journal_account_key.strip())

    any_token_configured = bool(SETTINGS.api_auth_token or SETTINGS.journal_auth_token)
    if not SETTINGS.is_deployed_environment and not any_token_configured:
        return local_access_context()

    if not token:
        auth_label = "Administrative" if required_role == "admin" else "Journal"
        raise HTTPException(
            status_code=401,
            detail=f"{auth_label} bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raise HTTPException(status_code=403, detail="Bearer token is invalid or does not have the required role.")


def require_admin(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(admin_auth),
) -> AccessContext:
    if SETTINGS.is_deployed_environment and not SETTINGS.api_auth_token:
        raise HTTPException(status_code=500, detail="API_AUTH_TOKEN is required for administrative API routes.")
    return access_from_request(request, credentials, "admin")


def require_journal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(admin_auth),
) -> AccessContext:
    if SETTINGS.is_deployed_environment and not SETTINGS.journal_auth_token:
        raise HTTPException(status_code=500, detail="JOURNAL_AUTH_TOKEN is required for server-side journal routes.")
    return access_from_request(request, credentials, "journal")


@router.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    return {"status": "ok", "requestId": request_id(), "database": database_summary(), "counts": current_counts()}


@router.get("/ready", response_model=ReadinessResponse)
def ready() -> dict[str, Any]:
    counts = current_counts()
    database_ready = counts["historical"] > 0 and counts["current"] > 0
    model_ready = False
    message = None
    if database_ready:
        try:
            load_serving_model_bundle()
            model_ready = True
        except HTTPException as exc:
            message = str(exc.detail)
    else:
        message = "Historical and current race data are required before the API is ready."
    return {
        "status": "ok" if database_ready and model_ready else "degraded",
        "requestId": request_id(),
        "databaseReady": database_ready,
        "modelReady": model_ready,
        "counts": counts,
        "message": message,
    }


@router.get("/summary", response_model=SummaryResponse)
def summary() -> dict[str, Any]:
    ensure_seed_data()
    counts = current_counts()
    status_df = ingestion_status(DATABASE_URL)
    last_refresh = None if status_df.empty else clean_value(status_df.iloc[0]["ingested_at"])
    return {
        "requestId": request_id(),
        "database": database_summary(),
        "historicalRuns": counts["historical"],
        "currentRunners": counts["current"],
        "lastRefresh": last_refresh,
        "dataFreshness": data_freshness(last_refresh),
    }


def quality_issue_payload(issue: Any) -> dict[str, Any]:
    return {
        "severity": issue.severity,
        "rowNumber": issue.row_number,
        "field": issue.field,
        "message": issue.message,
    }


@router.get("/data-quality", response_model=DataQualityResponse)
def data_quality() -> dict[str, Any]:
    ensure_seed_data()
    table_payloads = []
    for label, table_name, current_mode in [
        ("historical", TABLES["historical"], False),
        ("current", TABLES["current"], True),
    ]:
        frame = read_races(DATABASE_URL, table_name)
        enriched = enrich_race_frame(frame)
        issues = race_row_quality_issues(frame, current_mode=current_mode) + enrichment_quality_issues(frame)
        table_payloads.append(
            {
                "tableName": label,
                "rows": len(frame),
                "issueCount": len(issues),
                "issues": [quality_issue_payload(issue) for issue in issues[:25]],
                "coverage": field_coverage(enriched, ENRICHMENT_COLUMNS),
            }
        )
    return {"requestId": request_id(), "tables": table_payloads, "providerFreshness": provider_freshness_report(DATABASE_URL)}


def metric_payload(
    name: str,
    label: str,
    value: float | int | str | None,
    status: str = "ok",
    unit: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "value": clean_value(value),
        "unit": unit,
        "status": status,
        "description": description,
    }


def alert_payload(
    severity: str,
    category: str,
    code: str,
    message: str,
    value: float | int | str | None = None,
    threshold: float | int | str | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "category": category,
        "code": code,
        "message": message,
        "value": clean_value(value),
        "threshold": clean_value(threshold),
    }


def reference_prediction_frame(model: Any, history_features: pd.DataFrame) -> pd.DataFrame:
    reference_predictions = history_features.copy()
    if reference_predictions.empty:
        return reference_predictions

    probabilities = pd.Series(predict_probabilities(model, reference_predictions), index=reference_predictions.index)
    implied_probability = (
        pd.to_numeric(reference_predictions["implied_probability"], errors="coerce")
        if "implied_probability" in reference_predictions.columns
        else pd.Series(0.0, index=reference_predictions.index)
    )
    reference_predictions["win_probability"] = probabilities
    reference_predictions["model_odds"] = probabilities.map(lambda probability: 1 / probability if probability > 0 else None)
    reference_predictions["value_edge"] = probabilities - implied_probability
    return reference_predictions


def monitoring_status(drift_report: dict[str, Any], alerts: list[dict[str, Any]]) -> str:
    statuses = [str(drift_report.get("status") or "blocked")]
    statuses.extend(str(alert["severity"]) for alert in alerts)
    return worst_status(statuses)


def add_freshness_alerts(
    alerts: list[dict[str, Any]],
    freshness: dict[str, Any],
    provider_freshness: list[dict[str, Any]],
) -> None:
    freshness_status = freshness.get("status")
    if freshness_status in {"stale", "missing", "unknown"}:
        alerts.append(
            alert_payload(
                "critical" if freshness_status == "missing" else "warning",
                "freshness",
                f"data_freshness_{freshness_status}",
                f"Race data freshness is {freshness_status}.",
                value=freshness.get("ageHours"),
                threshold=freshness.get("maxAgeHours"),
            )
        )

    for row in provider_freshness:
        provider_status = str(row.get("status") or "unknown")
        if provider_status == "success":
            continue
        alerts.append(
            alert_payload(
                "critical" if provider_status == "failed" else "warning",
                "provider",
                f"provider_{provider_status}",
                f"{row.get('provider') or 'Provider'} {row.get('tableName') or 'table'} ingestion is {provider_status}.",
                value=row.get("rowCount"),
                threshold="success",
            )
        )


def add_runtime_alerts(alerts: list[dict[str, Any]], api_metrics: dict[str, Any], counts: dict[str, int]) -> None:
    if counts.get("historical", 0) == 0 or counts.get("current", 0) == 0:
        alerts.append(
            alert_payload(
                "critical",
                "data",
                "race_tables_empty",
                "Historical and current race tables must both contain rows before serving predictions.",
            )
        )

    if api_metrics.get("errorRate", 0) > SETTINGS.monitoring_max_error_rate:
        alerts.append(
            alert_payload(
                "critical",
                "api",
                "api_error_rate_high",
                "API error rate is above the configured monitoring threshold.",
                value=api_metrics.get("errorRate"),
                threshold=SETTINGS.monitoring_max_error_rate,
            )
        )

    if api_metrics.get("slowRequests", 0) > 0:
        alerts.append(
            alert_payload(
                "warning",
                "api",
                "api_slow_requests_seen",
                "At least one API request exceeded the configured slow-request threshold.",
                value=api_metrics.get("slowRequests"),
                threshold=SETTINGS.monitoring_slow_request_ms,
            )
        )


def build_monitoring_payload() -> dict[str, Any]:
    generated_at = utc_timestamp()
    ensure_seed_data()
    counts = current_counts()
    status_df = ingestion_status(DATABASE_URL)
    last_refresh = None if status_df.empty else clean_value(status_df.iloc[0]["ingested_at"])
    freshness = data_freshness(last_refresh)
    provider_freshness = provider_freshness_report(DATABASE_URL)
    api_metrics = api_metrics_snapshot()
    alerts: list[dict[str, Any]] = []

    history_df = read_races(DATABASE_URL, TABLES["historical"])
    current_df = read_races(DATABASE_URL, TABLES["current"])
    history_features = build_feature_table(history_df)
    try:
        current_features = add_scoring_features(current_df)
    except ValueError as exc:
        current_features = pd.DataFrame()
        alerts.append(
            alert_payload(
                "critical",
                "data",
                "current_feature_build_failed",
                f"Current feature generation failed: {exc}",
            )
        )

    reference_predictions: pd.DataFrame | None = None
    current_predictions: pd.DataFrame | None = None
    model_signal: dict[str, Any] = {
        "ready": False,
        "servingMode": "blocked",
        "modelVersionId": None,
        "artifactUri": None,
        "evaluationStatus": "skipped",
        "evaluationMessage": None,
        "tracingStatus": TRACING_STATUS,
    }

    try:
        _, _, model, _, model_record, serving_mode = load_serving_model_bundle()
        current_predictions = score_current_races(model, current_df)
        reference_predictions = reference_prediction_frame(model, history_features)
        model_signal.update(
            {
                "ready": True,
                "servingMode": serving_mode,
                "modelVersionId": model_record.get("id") if model_record else None,
                "artifactUri": model_record.get("artifactUri") if model_record else None,
                "trainingRows": model.training_rows,
                "featureCount": len(model.feature_columns),
            }
        )
    except HTTPException as exc:
        model_signal["evaluationMessage"] = str(exc.detail)
        alerts.append(
            alert_payload(
                "critical",
                "model",
                "model_serving_blocked",
                f"Model serving is blocked: {exc.detail}",
            )
        )

    evaluation = evaluate_model(history_features).to_dict()
    model_signal["evaluationStatus"] = evaluation["status"]
    model_signal["evaluationMessage"] = evaluation["message"]
    if evaluation["status"] != "ok":
        alerts.append(
            alert_payload(
                "warning",
                "model",
                "holdout_evaluation_not_ok",
                evaluation["message"] or "Holdout evaluation did not return an ok status.",
                value=evaluation["status"],
                threshold="ok",
            )
        )

    drift_report = build_drift_report(
        history_features,
        current_features,
        reference_predictions=reference_predictions,
        current_predictions=current_predictions,
        warning_threshold=SETTINGS.monitoring_drift_warning_threshold,
        critical_threshold=SETTINGS.monitoring_drift_critical_threshold,
    )
    alerts.extend(drift_alerts(drift_report))
    add_freshness_alerts(alerts, freshness, provider_freshness)
    add_runtime_alerts(alerts, api_metrics, counts)

    registry = read_model_registry(DATABASE_URL, limit=100)
    approved_models = [model for model in registry.get("models", []) if model.get("status") == "approved"]
    approved_artifacts = [model for model in approved_models if model.get("artifactReady")]
    if not approved_models:
        alerts.append(
            alert_payload(
                "warning",
                "governance",
                "no_approved_model",
                "No approved model version is recorded yet.",
                value=0,
                threshold=1,
            )
        )
    elif not approved_artifacts:
        alerts.append(
            alert_payload(
                "warning",
                "governance",
                "approved_model_artifact_missing",
                "An approved model exists but no approved artifact is ready.",
                value=0,
                threshold=1,
            )
        )

    runs = read_prediction_runs(DATABASE_URL, limit=1)
    if not runs.get("runs"):
        alerts.append(
            alert_payload(
                "warning",
                "governance",
                "no_prediction_run",
                "No persisted prediction run snapshot is recorded yet.",
                value=0,
                threshold=1,
            )
        )

    alert_rank = {"critical": 0, "blocked": 1, "warning": 2}
    alerts = sorted(alerts, key=lambda alert: (alert_rank.get(str(alert.get("severity")), 9), str(alert.get("code"))))
    metrics = [
        metric_payload("historical_rows", "Historical rows", counts.get("historical", 0), unit="count"),
        metric_payload("current_runners", "Current runners", counts.get("current", 0), unit="count"),
        metric_payload(
            "data_freshness",
            "Data freshness",
            freshness.get("ageHours"),
            status="ok" if freshness.get("status") == "fresh" else "warning",
            unit="hours",
            description=str(freshness.get("status") or "unknown"),
        ),
        metric_payload(
            "api_error_rate",
            "API error rate",
            api_metrics.get("errorRate"),
            status="critical" if api_metrics.get("errorRate", 0) > SETTINGS.monitoring_max_error_rate else "ok",
            unit="percent",
        ),
        metric_payload(
            "average_latency",
            "Average API latency",
            api_metrics.get("averageLatencyMs"),
            status="warning" if api_metrics.get("slowRequests", 0) > 0 else "ok",
            unit="ms",
        ),
        metric_payload(
            "feature_drift_checks",
            "Feature drift checks",
            len(drift_report.get("featureDrift", [])),
            status=drift_report.get("status", "blocked"),
            unit="count",
        ),
        metric_payload(
            "prediction_drift_checks",
            "Prediction drift checks",
            len(drift_report.get("predictionDrift", [])),
            status=drift_report.get("status", "blocked"),
            unit="count",
        ),
        metric_payload(
            "operator_alerts",
            "Operator alerts",
            len(alerts),
            status=worst_status([str(alert["severity"]) for alert in alerts]) if alerts else "ok",
            unit="count",
        ),
    ]

    return {
        "status": monitoring_status(drift_report, alerts),
        "generatedAt": generated_at,
        "dataFreshness": freshness,
        "providerFreshness": provider_freshness,
        "apiMetrics": api_metrics,
        "metrics": metrics,
        "alerts": alerts,
        "drift": drift_report,
        "model": model_signal,
    }


def audit_admin_action(
    access: AccessContext,
    action: str,
    resource_type: str,
    resource_id: str | int | None = None,
    status: str = "success",
    detail: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return record_admin_audit_event(
        DATABASE_URL,
        actor=access.actor,
        roles=access.roles,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        request_id=request_id(),
        status=status,
        detail=detail,
        payload=payload,
    )


@router.get("/monitoring", response_model=MonitoringResponse)
def monitoring() -> dict[str, Any]:
    return {"requestId": request_id(), **build_monitoring_payload()}


@router.get("/admin/session", response_model=AdminSessionResponse)
def admin_session(access: AccessContext | None = Depends(require_admin)) -> dict[str, Any]:
    admin = access_context_or_local(access, "admin")
    return access_payload(admin)


@router.get("/admin/audit-log", response_model=AdminAuditResponse)
def admin_audit_log(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    access: AccessContext | None = Depends(require_admin),
) -> dict[str, Any]:
    access_context_or_local(access, "admin")
    return {"requestId": request_id(), **read_admin_audit_events(DATABASE_URL, limit=limit, offset=offset)}


@router.get("/admin/governance", response_model=AdminGovernanceResponse)
def admin_governance(access: AccessContext | None = Depends(require_admin)) -> dict[str, Any]:
    admin = access_context_or_local(access, "admin")
    ingestion_payload = ingestion(limit=20)
    registry = read_model_registry(DATABASE_URL, limit=20)
    runs = read_prediction_runs(DATABASE_URL, limit=10)
    audit = read_admin_audit_events(DATABASE_URL, limit=10)
    return {
        "requestId": request_id(),
        "session": access_payload(admin),
        "readiness": ready(),
        "summary": summary(),
        "monitoring": monitoring(),
        "ingestion": ingestion_payload["ingestion"],
        "models": registry["models"],
        "predictionRuns": runs["runs"],
        "auditEvents": audit["events"],
    }


@router.get("/safeguards", response_model=ProductSafeguardsResponse)
def safeguards() -> dict[str, Any]:
    return {
        "requestId": request_id(),
        "responsibleUseNotice": "Horse Predictor is decision-support software, not betting advice or a guaranteed-return system.",
        "limitations": [
            "Predictions depend on provider coverage, data freshness, and historical data quality.",
            "Holdout metrics can drift when tracks, fields, weather, or provider schemas change.",
            "Value edges are model estimates and should be reviewed alongside bankroll controls and market context.",
        ],
        "dataLicensingNotice": SETTINGS.data_license_reference
        or "Only display race-card, odds, and result data that the operator is licensed to use.",
        "privacyNotice": "The bet journal stores operator-local records in the configured server database; do not enter personal data until account auth, export/delete controls, and a published privacy policy are configured.",
        "termsNotice": "Production launch requires published terms of use and no guaranteed-profit claims in product or marketing copy.",
        "links": {
            "responsibleGambling": SETTINGS.responsible_gambling_url or None,
            "privacyPolicy": SETTINGS.privacy_policy_url or None,
            "termsOfUse": SETTINGS.terms_of_use_url or None,
        },
    }


@router.get("/meetings", response_model=MeetingsResponse)
def meetings(
    race_date: Annotated[date | None, Query()] = None,
    track: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    ensure_seed_data()
    current_df = filter_race_rows(read_races(DATABASE_URL, TABLES["current"]), track=track, race_date=race_date)
    meeting_df = build_meetings(current_df)
    page_df, meta = page_frame(meeting_df, limit, offset)
    return {"requestId": request_id(), "meetings": records(page_df), "page": build_page(meta)}


@router.get("/races", response_model=RacesResponse)
def races(
    race_date: Annotated[date | None, Query()] = None,
    track: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    ensure_seed_data()
    current_df = filter_race_rows(read_races(DATABASE_URL, TABLES["current"]), track=track, race_date=race_date)
    race_df = build_race_summaries(current_df)
    page_df, meta = page_frame(race_df, limit, offset)
    return {"requestId": request_id(), "races": records(page_df), "page": build_page(meta)}


@router.get("/race-card", response_model=RaceCardResponse)
def race_card(
    race_date: Annotated[date | None, Query()] = None,
    track: Annotated[str | None, Query()] = None,
    horse: Annotated[str | None, Query()] = None,
    sort_by: Annotated[Literal["race_date", "track", "horse", "odds", "draw"], Query()] = "race_date",
    direction: Annotated[SortDirection, Query()] = "asc",
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    ensure_seed_data()
    current_df = filter_race_rows(read_races(DATABASE_URL, TABLES["current"]), track=track, race_date=race_date, horse=horse)
    current_df = sort_frame(current_df, sort_by, direction)
    page_df, meta = page_frame(current_df, limit, offset)
    return {"requestId": request_id(), "raceCard": records(page_df), "page": build_page(meta)}


@router.get("/predictions", response_model=PredictionsResponse)
def predictions(
    race_date: Annotated[date | None, Query()] = None,
    track: Annotated[str | None, Query()] = None,
    horse: Annotated[str | None, Query()] = None,
    sort_by: Annotated[Literal["suggested_rank", "win_probability", "value_edge", "odds", "horse", "track"], Query()] = "suggested_rank",
    direction: Annotated[SortDirection, Query()] = "asc",
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    scored_df = build_scored_predictions(race_date=race_date, track=track, horse=horse)
    scored_df = sort_frame(scored_df, sort_by, direction)
    page_df, meta = page_frame(scored_df, limit, offset)
    return {"requestId": request_id(), "predictions": records(page_df), "page": build_page(meta)}


@router.get("/prediction-runs", response_model=PredictionRunsResponse)
def prediction_runs(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    runs = read_prediction_runs(DATABASE_URL, limit=limit, offset=offset)
    return {"requestId": request_id(), **runs}


@router.get("/prediction-runs/{prediction_run_id}", response_model=PredictionRunResponse)
def prediction_run_detail(
    prediction_run_id: int,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    run = read_prediction_run(DATABASE_URL, prediction_run_id, limit=limit, offset=offset)
    if not run:
        raise HTTPException(status_code=404, detail=f"Prediction run {prediction_run_id} was not found.")
    return {"requestId": request_id(), **run}


@router.get("/bet-journal", response_model=BetJournalListResponse)
def bet_journal(
    status: Annotated[Literal["open", "won", "lost", "void"] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    access: AccessContext | None = Depends(require_journal),
) -> dict[str, Any]:
    journal_access = access_context_or_local(access, "journal")
    try:
        journal = read_bet_journal(
            DATABASE_URL,
            account_key=journal_access.account_key,
            status=status,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"requestId": request_id(), **journal}


@router.post("/bet-journal", response_model=BetJournalEntryResponse, status_code=http_status.HTTP_201_CREATED)
def create_bet(
    payload: BetJournalCreateRequest,
    access: AccessContext | None = Depends(require_journal),
) -> dict[str, Any]:
    journal_access = access_context_or_local(access, "journal")
    try:
        bet = record_bet_journal_entry(
            DATABASE_URL,
            request_model_payload(payload),
            account_key=journal_access.account_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"requestId": request_id(), "bet": bet}


@router.patch("/bet-journal/{bet_id}", response_model=BetJournalEntryResponse)
def update_bet(
    bet_id: int,
    payload: BetJournalUpdateRequest,
    access: AccessContext | None = Depends(require_journal),
) -> dict[str, Any]:
    journal_access = access_context_or_local(access, "journal")
    try:
        bet = update_bet_journal_entry(
            DATABASE_URL,
            bet_id,
            request_model_payload(payload),
            account_key=journal_access.account_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not bet:
        raise HTTPException(status_code=404, detail=f"Bet journal entry {bet_id} was not found.")
    return {"requestId": request_id(), "bet": bet}


@router.delete("/bet-journal/{bet_id}", response_model=BetJournalDeleteResponse)
def delete_bet(
    bet_id: int,
    access: AccessContext | None = Depends(require_journal),
) -> dict[str, Any]:
    journal_access = access_context_or_local(access, "journal")
    deleted = delete_bet_journal_entry(DATABASE_URL, bet_id, account_key=journal_access.account_key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Bet journal entry {bet_id} was not found.")
    return {"requestId": request_id(), "id": bet_id, "deleted": True}


@router.get("/entities/{entity_type}/{name}", response_model=EntityProfileResponse)
def entity_profile(
    entity_type: EntityType,
    name: str,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> dict[str, Any]:
    ensure_seed_data()
    history_df = read_races(DATABASE_URL, TABLES["historical"])
    matched = text_filter(history_df, entity_type, name, exact=True)
    if matched.empty:
        raise HTTPException(status_code=404, detail=f"No {entity_type} profile found for {name}.")

    finishing_position = pd.to_numeric(matched["finishing_position"], errors="coerce")
    wins = int((finishing_position == 1).sum())
    runs = int(len(matched))
    odds = pd.to_numeric(matched["odds"], errors="coerce")
    recent = matched.sort_values("race_date", ascending=False, kind="mergesort").head(limit)
    latest_date = pd.to_datetime(matched["race_date"], errors="coerce").max()
    return {
        "requestId": request_id(),
        "entityType": entity_type,
        "name": name,
        "runs": runs,
        "wins": wins,
        "winRate": wins / runs if runs else None,
        "averageOdds": float(odds.mean()) if not odds.dropna().empty else None,
        "latestRaceDate": clean_value(latest_date),
        "recentRuns": records(recent),
    }


@router.get("/model", response_model=ModelStatusResponse)
def model_status() -> dict[str, Any]:
    _, _, model, history_features, model_record, serving_mode = load_serving_model_bundle()
    evaluation = evaluate_model(history_features)
    return {
        "requestId": request_id(),
        "modelVersionId": model_record.get("id") if model_record else None,
        "servingMode": serving_mode,
        "artifactUri": model_record.get("artifactUri") if model_record else None,
        "trainingRows": model.training_rows,
        "winnerRate": model.winner_rate,
        "trainingStart": model.training_start,
        "trainingEnd": model.training_end,
        "featureCount": len(model.feature_columns),
        "features": model.feature_columns,
        "historicalRows": len(history_features),
        "evaluation": evaluation.to_dict(),
    }


@router.get("/model/evaluation", response_model=ModelEvaluationResponse)
def model_evaluation() -> dict[str, Any]:
    _, _, _, history_features = load_model_bundle()
    return {"requestId": request_id(), "evaluation": evaluate_model(history_features).to_dict()}


@router.get("/model/registry", response_model=ModelRegistryResponse)
def model_registry(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    registry = read_model_registry(DATABASE_URL, limit=limit, offset=offset)
    return {"requestId": request_id(), **registry}


@router.post("/admin/model/evaluation", response_model=ModelSnapshotResponse)
def capture_model_evaluation(access: AccessContext | None = Depends(require_admin)) -> dict[str, Any]:
    admin = access_context_or_local(access, "admin")
    _, _, model, history_features = load_model_bundle()
    evaluation = evaluate_model(history_features)
    artifact = save_model_artifact(model, SETTINGS.model_artifact_dir, code_commit_sha=current_code_commit_sha())
    model_record = record_model_evaluation_snapshot(
        DATABASE_URL,
        model,
        evaluation,
        artifact_uri=artifact.uri,
        artifact_sha256=artifact.sha256,
        feature_schema_hash=artifact.feature_schema_hash,
        code_commit_sha=artifact.code_commit_sha,
    )
    audit_admin_action(
        admin,
        "model_evaluation.capture",
        "model_version",
        model_record["id"],
        detail="Captured model evaluation snapshot and persisted model artifact.",
        payload={
            "status": model_record.get("status"),
            "artifactReady": model_record.get("artifactReady"),
            "trainingRows": model_record.get("metrics", {}).get("training_rows"),
            "featureCount": model_record.get("featureCount"),
        },
    )
    return {"requestId": request_id(), "model": model_record}


@router.post("/admin/model/{model_version_id}/approve", response_model=ModelSnapshotResponse)
def approve_model(
    model_version_id: int,
    access: AccessContext | None = Depends(require_admin),
) -> dict[str, Any]:
    admin = access_context_or_local(access, "admin")
    candidate = read_model_version(DATABASE_URL, model_version_id)
    if not candidate:
        raise HTTPException(status_code=404, detail=f"Model version {model_version_id} was not found.")
    if not candidate.get("artifactUri"):
        raise HTTPException(status_code=422, detail=f"Model version {model_version_id} has no persisted artifact to approve.")

    try:
        load_registry_model_artifact(candidate)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Model version {model_version_id} artifact failed integrity checks: {exc}") from exc

    model_record = approve_model_version(DATABASE_URL, model_version_id)
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model version {model_version_id} was not found.")
    audit_admin_action(
        admin,
        "model_version.approve",
        "model_version",
        model_version_id,
        detail="Approved model version for artifact-backed serving.",
        payload={
            "artifactReady": model_record.get("artifactReady"),
            "artifactSha256": model_record.get("artifactSha256"),
            "featureSchemaHash": model_record.get("featureSchemaHash"),
        },
    )
    return {"requestId": request_id(), "model": model_record}


@router.post("/admin/model/{model_version_id}/supersede", response_model=ModelSnapshotResponse)
def supersede_model(
    model_version_id: int,
    access: AccessContext | None = Depends(require_admin),
) -> dict[str, Any]:
    admin = access_context_or_local(access, "admin")
    model_record = update_model_version_status(DATABASE_URL, model_version_id, "superseded")
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model version {model_version_id} was not found.")
    audit_admin_action(
        admin,
        "model_version.supersede",
        "model_version",
        model_version_id,
        detail="Superseded model version from the admin console.",
        payload={"previousStatus": "unknown", "status": model_record.get("status")},
    )
    return {"requestId": request_id(), "model": model_record}


@router.post("/admin/prediction-runs", response_model=PredictionRunResponse)
def capture_prediction_run(
    access: AccessContext | None = Depends(require_admin),
    race_date: Annotated[date | None, Query()] = None,
    track: Annotated[str | None, Query()] = None,
    horse: Annotated[str | None, Query()] = None,
    require_approved_model: Annotated[bool, Query()] = False,
) -> dict[str, Any]:
    admin = access_context_or_local(access, "admin")
    scored_df, model_record, serving_mode = build_scored_predictions_with_model(race_date=race_date, track=track, horse=horse)
    scored_df = sort_frame(scored_df, "suggested_rank", "asc")
    if scored_df.empty:
        raise HTTPException(status_code=422, detail="No predictions matched the requested snapshot filters.")

    if require_approved_model and (not model_record or serving_mode != "artifact"):
        raise HTTPException(status_code=422, detail="An approved model artifact is required before recording this prediction run.")

    model_version_id = int(model_record["id"]) if model_record else None
    notes = (
        "scored with approved model artifact"
        if model_record
        else "scored with current in-memory model; no approved model artifact linked"
    )
    run = record_prediction_run(DATABASE_URL, scored_df, source="api", model_version_id=model_version_id, notes=notes)
    audit_admin_action(
        admin,
        "prediction_run.capture",
        "prediction_run",
        run["run"]["id"],
        detail="Captured persisted prediction run snapshot.",
        payload={
            "modelVersionId": model_version_id,
            "servingMode": serving_mode,
            "runnerCount": run["run"].get("runnerCount"),
            "requireApprovedModel": require_approved_model,
            "filters": {
                "raceDate": clean_value(race_date),
                "track": track,
                "horse": horse,
            },
        },
    )
    return {"requestId": request_id(), **run}


@router.get("/trends", response_model=TrendsResponse)
def trends(limit: Annotated[int, Query(ge=1, le=100)] = 25) -> dict[str, Any]:
    _, _, _, history_features = load_model_bundle()
    entity_tables = summarize_entities(history_features)
    return {"requestId": request_id(), **{name: records(table.head(limit)) for name, table in entity_tables.items()}}


@router.get("/ingestion-status", response_model=IngestionStatusResponse)
def ingestion(limit: Annotated[int, Query(ge=1, le=100)] = 20, offset: Annotated[int, Query(ge=0)] = 0) -> dict[str, Any]:
    ensure_seed_data()
    status_df = ingestion_status(DATABASE_URL)
    page_df, meta = page_frame(status_df, limit, offset)
    return {"requestId": request_id(), "ingestion": records(page_df), "page": build_page(meta)}


@router.post("/admin/seed-sample", response_model=SeedSampleResponse)
def seed_sample(access: AccessContext | None = Depends(require_admin)) -> dict[str, Any]:
    admin = access_context_or_local(access, "admin")
    seeded = seed_database_from_samples(DATABASE_URL, SETTINGS.sample_historical_csv, SETTINGS.sample_current_csv)
    audit_admin_action(
        admin,
        "data.seed_sample",
        "database",
        "sample-data",
        detail="Seeded sample historical and current race tables.",
        payload={"seeded": seeded},
    )
    return {
        "requestId": request_id(),
        "seeded": seeded,
    }


app.include_router(router, prefix="/api/v1")
app.include_router(router, prefix="/api")
