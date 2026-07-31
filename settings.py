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


@dataclass(frozen=True)
class Settings:
    app_env: str
    database_url: str
    backend_cors_origins: tuple[str, ...]
    horse_api_provider: str
    horse_api_base_url: str
    horse_api_key: str
    racing_api_username: str
    racing_api_password: str
    vite_api_base_url: str
    sample_historical_csv: Path
    sample_current_csv: Path
    api_auth_token: str = ""

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
        if "*" in self.backend_cors_origins:
            errors.append("BACKEND_CORS_ORIGINS must not use '*' with credentials enabled.")

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
    default_origins = ("http://localhost:5173", "http://127.0.0.1:5173")
    return Settings(
        app_env=os.getenv("APP_ENV", "development").lower(),
        database_url=resolve_database_url(),
        backend_cors_origins=_split_csv(os.getenv("BACKEND_CORS_ORIGINS", ""), default_origins),
        horse_api_provider=os.getenv("HORSE_API_PROVIDER", "sample"),
        horse_api_base_url=os.getenv("HORSE_API_BASE_URL", ""),
        horse_api_key=os.getenv("HORSE_API_KEY", ""),
        racing_api_username=os.getenv("RACING_API_USERNAME", ""),
        racing_api_password=os.getenv("RACING_API_PASSWORD", ""),
        vite_api_base_url=os.getenv("VITE_API_BASE_URL", "http://127.0.0.1:8000"),
        sample_historical_csv=Path(os.getenv("SAMPLE_HISTORICAL_CSV", BASE_DIR / "sample_historical_data.csv")),
        sample_current_csv=Path(os.getenv("SAMPLE_CURRENT_CSV", BASE_DIR / "sample_current_races.csv")),
        api_auth_token=os.getenv("API_AUTH_TOKEN", ""),
    )
