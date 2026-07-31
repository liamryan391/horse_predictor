from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from data_quality import model_quality_issues, race_row_quality_issues
from prediction_model import build_feature_table, evaluate_model


ROOT = Path(__file__).resolve().parents[1]


class DataQualityTests(unittest.TestCase):
    def test_sample_files_pass_required_data_quality_gates(self) -> None:
        historical = pd.read_csv(ROOT / "sample_historical_data.csv")
        current = pd.read_csv(ROOT / "sample_current_races.csv")

        self.assertEqual([], race_row_quality_issues(historical))
        self.assertEqual([], race_row_quality_issues(current, current_mode=True))

    def test_invalid_rows_report_required_odds_result_and_duplicate_issues(self) -> None:
        historical = pd.read_csv(ROOT / "sample_historical_data.csv").head(1)
        broken = pd.concat([historical, historical, historical], ignore_index=True)
        broken.loc[0, "horse"] = ""
        broken.loc[0, "odds"] = 1.0
        broken.loc[0, "finishing_position"] = 0

        fields = {issue.field for issue in race_row_quality_issues(broken)}

        self.assertIn("horse", fields)
        self.assertIn("odds", fields)
        self.assertIn("finishing_position", fields)
        self.assertIn("race_identity", fields)

    def test_model_quality_gates_require_market_baselines_and_bounded_calibration(self) -> None:
        historical = pd.read_csv(ROOT / "sample_historical_data.csv")
        evaluation = evaluate_model(build_feature_table(historical))

        self.assertEqual([], model_quality_issues(evaluation))

        degraded = {
            "status": "ok",
            "metrics": {
                "runner_brier_score": 0.6,
                "market_brier_score": 0.3,
                "runner_log_loss": 1.4,
                "market_log_loss": 1.0,
                "calibration_mae": 0.9,
            },
        }
        fields = {issue.field for issue in model_quality_issues(degraded)}

        self.assertEqual({"calibration_mae", "runner_brier_score", "runner_log_loss"}, fields)


if __name__ == "__main__":
    unittest.main()
