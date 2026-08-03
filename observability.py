from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field_name in [
            "request_id",
            "method",
            "path",
            "elapsed_ms",
            "status_code",
            "slow_request",
            "provider",
            "target_table",
            "metric",
            "value",
            "threshold",
            "severity",
            "category",
            "otel_status",
            "service_name",
        ]:
            if hasattr(record, field_name):
                payload[field_name] = getattr(record, field_name)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(app_env: str, log_format: str) -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()

    if log_format == "json" or app_env in {"staging", "production"}:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]


def configure_tracing(app: Any, enabled: bool, service_name: str) -> str:
    if not enabled:
        return "disabled"

    tracer_logger = logging.getLogger("horse_predictor.observability")
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        tracer_logger.warning(
            "otel_unavailable",
            extra={"otel_status": "unavailable", "service_name": service_name},
        )
        return "unavailable"

    try:
        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        tracer_logger.exception(
            "otel_instrumentation_failed",
            extra={"otel_status": "failed", "service_name": service_name},
        )
        return "failed"

    tracer_logger.info(
        "otel_instrumentation_enabled",
        extra={"otel_status": "enabled", "service_name": service_name},
    )
    return "enabled"
