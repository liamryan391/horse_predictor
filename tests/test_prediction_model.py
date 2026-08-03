from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from prediction_model import (
    assert_no_leakage,
    build_feature_table,
    evaluate_model,
    leakage_features,
    load_model_artifact,
    save_model_artifact,
    score_current_races,
    train_model,
)


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
        self.assertIn("country", model.feature_columns)
        self.assertIn("distance_bucket", model.feature_columns)
        self.assertIn("going_category", model.feature_columns)
        self.assertIn("race_type", model.feature_columns)
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

        for column in [
            "country",
            "distance_bucket",
            "going_category",
            "race_type",
            "field_size",
            "odds_rank",
            "relative_speed_rating",
            "relative_class_rating",
        ]:
            self.assertIn(column, features.columns)

        ascot_rows = features[features["track"] == "Ascot"]
        self.assertEqual([2, 2], ascot_rows["field_size"].tolist())
        self.assertEqual(["GB"], ascot_rows["country"].dropna().unique().tolist())

    def test_model_artifact_roundtrip_keeps_predictions_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            features = build_feature_table(self.history)
            current = pd.read_csv(ROOT / "sample_current_races.csv")
            model = train_model(features)
            expected = score_current_races(model, current)["win_probability"].round(12).tolist()

            artifact = save_model_artifact(model, temp_dir, code_commit_sha="unit-test")
            loaded = load_model_artifact(
                artifact.uri,
                expected_sha256=artifact.sha256,
                expected_feature_schema_hash=artifact.feature_schema_hash,
            )
            actual = score_current_races(loaded, current)["win_probability"].round(12).tolist()

            self.assertEqual(expected, actual)
            self.assertEqual("unit-test", artifact.code_commit_sha)
            self.assertEqual(64, len(artifact.sha256))

            with self.assertRaises(ValueError):
                load_model_artifact(artifact.uri, expected_sha256="0" * 64)

            with self.assertRaises(ValueError):
                load_model_artifact(artifact.uri, expected_feature_schema_hash="1" * 64)


if __name__ == "__main__":
    unittest.main()
