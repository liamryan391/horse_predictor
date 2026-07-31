from __future__ import annotations

import contextvars
import logging
import math
import re
import time
import uuid
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from secrets import compare_digest
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.trustedhost import TrustedHostMiddleware

from observability import configure_logging
from api_contracts import (
    EntityProfileResponse,
    ErrorResponse,
    HealthResponse,
    IngestionStatusResponse,
    MeetingsResponse,
    ModelEvaluationResponse,
    ModelStatusResponse,
    PageMeta,
    PredictionsResponse,
    ProductSafeguardsResponse,
    RaceCardResponse,
    RacesResponse,
    ReadinessResponse,
    SeedSampleResponse,
    SummaryResponse,
    TrendsResponse,
)
from prediction_model import build_feature_table, evaluate_model, score_current_races, summarize_entities, train_model
from racing_storage import (
    TABLES,
    ingestion_status,
    read_races,
    seed_database_from_samples,
    table_counts,
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

app = FastAPI(
    title="Horse Predictor API",
    version="0.3.0",
    description="Versioned API for race cards, model predictions, ingestion status, and model evaluation.",
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(SETTINGS.allowed_hosts))
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.backend_cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

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
        response.headers["X-Request-ID"] = active_request_id
        apply_security_headers(response)
        return response
    finally:
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)
        logger.info(
            "api_request",
            extra={
                "request_id": active_request_id,
                "method": request.method,
                "path": request.url.path,
                "elapsed_ms": elapsed_ms,
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


def current_counts() -> dict[str, int]:
    return table_counts(DATABASE_URL)


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


def require_admin(credentials: HTTPAuthorizationCredentials | None = Depends(admin_auth)) -> None:
    if not SETTINGS.api_auth_token:
        if SETTINGS.is_deployed_environment:
            raise HTTPException(status_code=500, detail="API_AUTH_TOKEN is required for administrative API routes.")
        return
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Administrative token required.", headers={"WWW-Authenticate": "Bearer"})
    if not compare_digest(credentials.credentials, SETTINGS.api_auth_token):
        raise HTTPException(status_code=403, detail="Administrative token is invalid.")


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
            load_model_bundle()
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
        "privacyNotice": "The current bet journal stores entries in browser local storage; do not enter personal data until account storage and a published privacy policy are configured.",
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
    _, current_df, model, _ = load_model_bundle()
    scored_df = score_current_races(model, current_df)
    scored_df = filter_race_rows(scored_df, track=track, race_date=race_date, horse=horse)
    scored_df = sort_frame(scored_df, sort_by, direction)
    page_df, meta = page_frame(scored_df, limit, offset)
    return {"requestId": request_id(), "predictions": records(page_df), "page": build_page(meta)}


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
    _, _, model, history_features = load_model_bundle()
    evaluation = evaluate_model(history_features)
    return {
        "requestId": request_id(),
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
def seed_sample(_: None = Depends(require_admin)) -> dict[str, Any]:
    return {
        "requestId": request_id(),
        "seeded": seed_database_from_samples(DATABASE_URL, SETTINGS.sample_historical_csv, SETTINGS.sample_current_csv),
    }


app.include_router(router, prefix="/api/v1")
app.include_router(router, prefix="/api")
