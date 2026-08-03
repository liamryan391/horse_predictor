from __future__ import annotations

import unittest

import pandas as pd

from monitoring import api_metrics_snapshot, build_drift_report, record_api_request, reset_api_metrics


class MonitoringTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_api_metrics()

    def test_api_metrics_capture_errors_latency_and_path_counts(self) -> None:
        reset_api_metrics()

        record_api_request("GET", "/api/v1/health", 200, 12.5, slow_request_ms=50)
        record_api_request("GET", "/api/v1/model", 503, 88.0, slow_request_ms=50)

        snapshot = api_metrics_snapshot()

        self.assertEqual(2, snapshot["totalRequests"])
        self.assertEqual(1, snapshot["errorRequests"])
        self.assertEqual(1, snapshot["slowRequests"])
        self.assertEqual(0.5, snapshot["errorRate"])
        self.assertEqual({"200": 1, "503": 1}, snapshot["statusCounts"])
        self.assertEqual("/api/v1/health", snapshot["recent"][0]["path"])
        self.assertIsNotNone(snapshot["lastErrorAt"])
        self.assertIsNotNone(snapshot["lastSlowAt"])

    def test_drift_report_flags_feature_and_prediction_changes(self) -> None:
        reference = pd.DataFrame(
            {
                "odds": [2.0, 3.0, 4.0],
                "track": ["York", "York", "Ascot"],
                "speed_rating": [70, 72, 74],
                "win_probability": [0.2, 0.3, 0.4],
                "model_odds": [5.0, 3.33, 2.5],
                "value_edge": [0.01, 0.02, 0.03],
            }
        )
        current = pd.DataFrame(
            {
                "odds": [12.0, 13.0, 14.0],
                "track": ["Kempton", "Kempton", "Kempton"],
                "speed_rating": [55, 56, 57],
                "win_probability": [0.75, 0.8, 0.85],
                "model_odds": [1.33, 1.25, 1.18],
                "value_edge": [0.5, 0.55, 0.6],
            }
        )

        report = build_drift_report(
            reference,
            current,
            reference_predictions=reference,
            current_predictions=current,
            warning_threshold=0.2,
            critical_threshold=0.5,
        )
        feature_rows = {row["field"]: row for row in report["featureDrift"]}
        prediction_rows = {row["field"]: row for row in report["predictionDrift"]}

        self.assertEqual("critical", report["status"])
        self.assertEqual("critical", feature_rows["odds"]["status"])
        self.assertEqual(["Kempton"], feature_rows["track"]["newCategories"])
        self.assertEqual("critical", prediction_rows["win_probability"]["status"])


if __name__ == "__main__":
    unittest.main()
