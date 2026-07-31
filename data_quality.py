from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from schema import RACE_COLUMNS, RACE_IDENTITY_COLUMNS


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    row_number: int | None
    field: str
    message: str


def required_column_issues(df: pd.DataFrame, required_columns: Iterable[str] = RACE_COLUMNS) -> list[QualityIssue]:
    return [
        QualityIssue("error", None, column, "Required column is missing.")
        for column in required_columns
        if column not in df.columns
    ]


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def _append_row_issues(
    issues: list[QualityIssue],
    mask: pd.Series,
    field: str,
    message: str,
    severity: str = "error",
) -> None:
    for row_number in mask[mask].index.tolist():
        issues.append(QualityIssue(severity, int(row_number), field, message))


def race_row_quality_issues(df: pd.DataFrame, current_mode: bool = False) -> list[QualityIssue]:
    issues = required_column_issues(df)
    if issues:
        return issues

    required_fields = ["race_date", "track", "horse", "odds"]
    if not current_mode:
        required_fields.append("finishing_position")

    for field in required_fields:
        _append_row_issues(issues, _missing_mask(df[field]), field, "Required field is missing or blank.")

    odds = pd.to_numeric(df["odds"], errors="coerce")
    _append_row_issues(issues, odds.isna() | (odds <= 1), "odds", "Decimal odds must be greater than 1.")

    positions = pd.to_numeric(df["finishing_position"], errors="coerce")
    if not current_mode:
        invalid_positions = positions.isna() | (positions < 1) | (positions % 1 != 0)
        _append_row_issues(
            issues,
            invalid_positions,
            "finishing_position",
            "Historical finishing position must be a positive whole number.",
        )

    identity_columns = [column for column in RACE_IDENTITY_COLUMNS if column in df.columns]
    if identity_columns:
        identity_frame = df[identity_columns].fillna("__missing__").astype(str)
        duplicate_mask = identity_frame.duplicated(keep=False)
        _append_row_issues(issues, duplicate_mask, "race_identity", "Duplicate race/runner identity found.")

    return issues


def model_quality_issues(evaluation: Any, max_calibration_mae: float = 0.75) -> list[QualityIssue]:
    status = getattr(evaluation, "status", None) if not isinstance(evaluation, dict) else evaluation.get("status")
    metrics = getattr(evaluation, "metrics", None) if not isinstance(evaluation, dict) else evaluation.get("metrics")
    metrics = metrics or {}
    issues: list[QualityIssue] = []

    if status != "ok":
        issues.append(QualityIssue("error", None, "evaluation_status", "Model evaluation is not ready."))

    required_metrics = [
        "runner_brier_score",
        "market_brier_score",
        "runner_log_loss",
        "market_log_loss",
        "calibration_mae",
    ]
    for metric in required_metrics:
        if metrics.get(metric) is None:
            issues.append(QualityIssue("error", None, metric, "Required model metric is missing."))

    calibration_mae = metrics.get("calibration_mae")
    if calibration_mae is not None and calibration_mae > max_calibration_mae:
        issues.append(QualityIssue("warning", None, "calibration_mae", "Calibration gap exceeds threshold."))

    runner_brier = metrics.get("runner_brier_score")
    market_brier = metrics.get("market_brier_score")
    if runner_brier is not None and market_brier is not None and runner_brier > market_brier:
        issues.append(QualityIssue("warning", None, "runner_brier_score", "Model Brier score is worse than market."))

    runner_log_loss = metrics.get("runner_log_loss")
    market_log_loss = metrics.get("market_log_loss")
    if runner_log_loss is not None and market_log_loss is not None and runner_log_loss > market_log_loss:
        issues.append(QualityIssue("warning", None, "runner_log_loss", "Model log loss is worse than market."))

    return issues
