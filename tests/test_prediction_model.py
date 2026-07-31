from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from prediction_model import assert_no_leakage, build_feature_table, evaluate_model, leakage_features, train_model


ROOT = Path(__file__).resolve().parents[1]


class PredictionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = pd.read_csv(ROOT / "sample_historical_data.csv")

    def test_train_model_excludes_result_only_columns(self) -> None:
        features = build_feature_table(self.history)
        model = train_model(features)

        self.assertNotIn("finishing_position", model.feature_columns)
        self.assertNotIn("is_winner", model.feature_columns)
        self.assertEqual([], leakage_features(model.feature_columns))

        with self.assertRaises(ValueError):
            assert_no_leakage(["odds", "is_winner"])

    def test_chronological_holdout_uses_latest_race(self) -> None:
        features = build_feature_table(self.history)
        evaluation = evaluate_model(features, validation_fraction=0.25)

        self.assertEqual("ok", evaluation.status)
        self.assertEqual(2, evaluation.training_races)
        self.assertEqual(1, evaluation.validation_races)
        self.assertEqual("2025-03-18", evaluation.evaluation_start)
        self.assertEqual("2025-03-18", evaluation.evaluation_end)
        self.assertIn("runner_brier_score", evaluation.metrics)

    def test_feature_table_adds_pre_race_context_features(self) -> None:
        features = build_feature_table(self.history)

        for column in ["field_size", "odds_rank", "relative_speed_rating", "relative_class_rating"]:
            self.assertIn(column, features.columns)

        ascot_rows = features[features["track"] == "Ascot"]
        self.assertEqual([2, 2], ascot_rows["field_size"].tolist())


if __name__ == "__main__":
    unittest.main()
