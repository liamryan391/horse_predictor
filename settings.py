from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url

BASE_DIR = Path(__file__).resolve().parent
LOCAL_ENV_FILE = BASE_DIR / ".env"

load_dotenv(LOCAL_ENV_FILE, override=False)


def _split_csv(value: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or fallback


def _bool_env(value: str | None, fallback: bool) -> bool:
    if value is None or value == "":
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str
    database_url: str
    backend_cors_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    horse_api_provider: str
    horse_api_base_url: str
    horse_api_key: str
    racing_api_username: str
    racing_api_password: str
    horse_api_timeout_seconds: float
    horse_api_retry_attempts: int
    horse_api_retry_backoff_seconds: float
    horse_api_min_request_interval_seconds: float
    horse_api_max_pages: int
    vite_api_base_url: str
    sample_historical_csv: Path
    sample_current_csv: Path
    api_auth_token: str = ""
    journal_auth_token: str = ""
    journal_account_key: str = "journal-user"
    account_auth_enabled: bool = True
    api_rate_limit_per_minute: int = 240
    data_freshness_max_age_hours: float = 24.0
    log_format: str = "plain"
    max_request_body_bytes: int = 1_048_576
    monitoring_drift_warning_threshold: float = 0.35
    monitoring_drift_critical_threshold: float = 0.75
    monitoring_slow_request_ms: float = 1000.0
    monitoring_max_error_rate: float = 0.05
    otel_enabled: bool = False
    otel_service_name: str = "horse-predictor-api"
    responsible_gambling_url: str = ""
    privacy_policy_url: str = ""
    terms_of_use_url: str = ""
    data_license_reference: str = ""
    model_artifact_dir: Path = BASE_DIR / "model_artifacts"
    require_approved_model_artifact: bool = False
    broker_raw_cache_dir: Path = BASE_DIR / "broker_payloads"
    ai_provider: str = "disabled"
    ai_base_url: str = "http://127.0.0.1:11434/v1"
    ai_model: str = ""
    ai_timeout_seconds: float = 20.0
    ai_json_schema_required: bool = True
    ai_api_key: str = ""

    @property
    def is_deployed_environment(self) -> bool:
        return self.app_env in {"staging", "production"}

    def validate_runtime(self) -> None:
        errors: list[str] = []
        parsed = make_url(resolve_database_url(self.database_url))

        if self.is_deployed_environment and parsed.get_backend_name() == "sqlite":
            errors.append("DATABASE_URL must use MySQL or another managed SQL database outside local development.")
        if self.is_deployed_environment and not self.backend_cors_origins:
            errors.append("BACKEND_CORS_ORIGINS must list the deployed frontend origins.")
        if self.is_deployed_environment and not self.allowed_hosts:
            errors.append("ALLOWED_HOSTS must list the deployed API hostnames.")
        if "*" in self.backend_cors_origins:
            errors.append("BACKEND_CORS_ORIGINS must not use '*' with credentials enabled.")
        if "*" in self.allowed_hosts and self.is_deployed_environment:
            errors.append("ALLOWED_HOSTS must not use '*' in staging or production.")
        if self.is_deployed_environment:
            insecure_origins = [origin for origin in self.backend_cors_origins if origin.startswith("http://")]
            if insecure_origins:
                errors.append("BACKEND_CORS_ORIGINS must use HTTPS in staging or production.")
        if self.is_deployed_environment and not self.account_auth_enabled and not self.api_auth_token:
            errors.append("API_AUTH_TOKEN must be set before enabling administrative API routes without account auth.")
        if self.is_deployed_environment and self.api_auth_token:
            weak_tokens = {"replace-with-a-long-random-token", "replace_me", "changeme", "change-me"}
            token_value = self.api_auth_token.lower()
            if len(self.api_auth_token) < 32 or token_value in weak_tokens or "replace" in token_value:
                errors.append("API_AUTH_TOKEN must be a non-placeholder secret with at least 32 characters.")
        if self.is_deployed_environment and not self.account_auth_enabled and not self.journal_auth_token:
            errors.append("JOURNAL_AUTH_TOKEN must be set before enabling server-side journal routes without account auth.")
        if self.is_deployed_environment and self.journal_auth_token:
            weak_tokens = {"replace-with-a-long-random-token", "replace_me", "changeme", "change-me"}
            token_value = self.journal_auth_token.lower()
            if len(self.journal_auth_token) < 32 or token_value in weak_tokens or "replace" in token_value:
                errors.append("JOURNAL_AUTH_TOKEN must be a non-placeholder secret with at least 32 characters.")
        if not self.journal_account_key.strip():
            errors.append("JOURNAL_ACCOUNT_KEY must not be empty.")
        if self.max_request_body_bytes < 1024:
            errors.append("MAX_REQUEST_BODY_BYTES must be at least 1024.")
        if not 0 < self.monitoring_drift_warning_threshold <= self.monitoring_drift_critical_threshold <= 1:
            errors.append("Monitoring drift thresholds must satisfy 0 < warning <= critical <= 1.")
        if self.monitoring_slow_request_ms <= 0:
            errors.append("MONITORING_SLOW_REQUEST_MS must be greater than 0.")
        if not 0 <= self.monitoring_max_error_rate <= 1:
            errors.append("MONITORING_MAX_ERROR_RATE must be between 0 and 1.")
        if self.require_approved_model_artifact and not self.model_artifact_dir:
            errors.append("MODEL_ARTIFACT_DIR must be configured when approved model artifacts are required.")
        if not str(self.broker_raw_cache_dir).strip():
            errors.append("BROKER_RAW_CACHE_DIR must not be empty.")
        ai_enabled = self.ai_provider.strip().lower() not in {"", "disabled", "off", "none"}
        if ai_enabled:
            if not self.ai_base_url.strip():
                errors.append("AI_BASE_URL must be set when AI_PROVIDER is enabled.")
            if not self.ai_model.strip():
                errors.append("AI_MODEL must be set when AI_PROVIDER is enabled.")
            if self.is_deployed_environment and self.ai_base_url.startswith("http://"):
                loopback_prefixes = ("http://127.0.0.1", "http://localhost")
                if not self.ai_base_url.startswith(loopback_prefixes):
                    errors.append("AI_BASE_URL must use HTTPS outside loopback in staging or production.")
        if self.ai_timeout_seconds <= 0:
            errors.append("AI_TIMEOUT_SECONDS must be greater than 0.")

        if errors:
            raise RuntimeError("Invalid Horse Predictor configuration: " + " ".join(errors))

    def database_summary(self) -> dict[str, Any]:
        parsed = make_url(resolve_database_url(self.database_url))
        database_name = parsed.database
        if parsed.get_backend_name() == "sqlite" and database_name:
            database_name = Path(database_name).name
        return {
            "environment": self.app_env,
            "engine": parsed.get_backend_name(),
            "driver": parsed.drivername,
            "database": database_name,
        }


def resolve_database_url(database_url: str | Path | None = None) -> str:
    raw_url = str(database_url or os.getenv("DATABASE_URL") or os.getenv("HORSE_DB_PATH") or BASE_DIR / "horse_racing.db")
    if "://" in raw_url:
        return raw_url
    return f"sqlite:///{Path(raw_url).as_posix()}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    app_env = os.getenv("APP_ENV", "development").lower()
    deployed = app_env in {"staging", "production"}
    default_origins = ("http://localhost:5173", "http://127.0.0.1:5173")
    default_allowed_hosts = ("localhost", "127.0.0.1", "testserver")
    return Settings(
        app_env=app_env,
        database_url=resolve_database_url(),
        backend_cors_origins=_split_csv(os.getenv("BACKEND_CORS_ORIGINS", ""), () if deployed else default_origins),
        allowed_hosts=_split_csv(os.getenv("ALLOWED_HOSTS", ""), () if deployed else default_allowed_hosts),
        horse_api_provider=os.getenv("HORSE_API_PROVIDER", "sample"),
        horse_api_base_url=os.getenv("HORSE_API_BASE_URL", ""),
        horse_api_key=os.getenv("HORSE_API_KEY", ""),
        racing_api_username=os.getenv("RACING_API_USERNAME", ""),
        racing_api_password=os.getenv("RACING_API_PASSWORD", ""),
        horse_api_timeout_seconds=float(os.getenv("HORSE_API_TIMEOUT_SECONDS", "30")),
        horse_api_retry_attempts=int(os.getenv("HORSE_API_RETRY_ATTEMPTS", "3")),
        horse_api_retry_backoff_seconds=float(os.getenv("HORSE_API_RETRY_BACKOFF_SECONDS", "1.5")),
        horse_api_min_request_interval_seconds=float(os.getenv("HORSE_API_MIN_REQUEST_INTERVAL_SECONDS", "0")),
        horse_api_max_pages=int(os.getenv("HORSE_API_MAX_PAGES", "5")),
        vite_api_base_url=os.getenv("VITE_API_BASE_URL", "http://127.0.0.1:8000"),
        sample_historical_csv=Path(os.getenv("SAMPLE_HISTORICAL_CSV", BASE_DIR / "sample_historical_data.csv")),
        sample_current_csv=Path(os.getenv("SAMPLE_CURRENT_CSV", BASE_DIR / "sample_current_races.csv")),
        api_auth_token=os.getenv("API_AUTH_TOKEN", ""),
        journal_auth_token=os.getenv("JOURNAL_AUTH_TOKEN", ""),
        journal_account_key=os.getenv("JOURNAL_ACCOUNT_KEY", "journal-user"),
        account_auth_enabled=_bool_env(os.getenv("ACCOUNT_AUTH_ENABLED"), True),
        api_rate_limit_per_minute=int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "240")),
        data_freshness_max_age_hours=float(os.getenv("DATA_FRESHNESS_MAX_AGE_HOURS", "24")),
        log_format=os.getenv("LOG_FORMAT", "json" if deployed else "plain"),
        max_request_body_bytes=int(os.getenv("MAX_REQUEST_BODY_BYTES", str(1_048_576))),
        monitoring_drift_warning_threshold=float(os.getenv("MONITORING_DRIFT_WARNING_THRESHOLD", "0.35")),
        monitoring_drift_critical_threshold=float(os.getenv("MONITORING_DRIFT_CRITICAL_THRESHOLD", "0.75")),
        monitoring_slow_request_ms=float(os.getenv("MONITORING_SLOW_REQUEST_MS", "1000")),
        monitoring_max_error_rate=float(os.getenv("MONITORING_MAX_ERROR_RATE", "0.05")),
        otel_enabled=_bool_env(os.getenv("OTEL_ENABLED"), False),
        otel_service_name=os.getenv("OTEL_SERVICE_NAME", "horse-predictor-api"),
        responsible_gambling_url=os.getenv("RESPONSIBLE_GAMBLING_URL", ""),
        privacy_policy_url=os.getenv("PRIVACY_POLICY_URL", ""),
        terms_of_use_url=os.getenv("TERMS_OF_USE_URL", ""),
        data_license_reference=os.getenv("DATA_LICENSE_REFERENCE", ""),
        model_artifact_dir=Path(os.getenv("MODEL_ARTIFACT_DIR", BASE_DIR / "model_artifacts")),
        require_approved_model_artifact=_bool_env(os.getenv("REQUIRE_APPROVED_MODEL_ARTIFACT"), deployed),
        broker_raw_cache_dir=Path(os.getenv("BROKER_RAW_CACHE_DIR", BASE_DIR / "broker_payloads")),
        ai_provider=os.getenv("AI_PROVIDER", "disabled").lower(),
        ai_base_url=os.getenv("AI_BASE_URL", "http://127.0.0.1:11434/v1"),
        ai_model=os.getenv("AI_MODEL", ""),
        ai_timeout_seconds=float(os.getenv("AI_TIMEOUT_SECONDS", "20")),
        ai_json_schema_required=_bool_env(os.getenv("AI_JSON_SCHEMA_REQUIRED"), True),
        ai_api_key=os.getenv("AI_API_KEY", ""),
    )
