from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

_TEMP_DIR = tempfile.TemporaryDirectory()
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(_TEMP_DIR.name) / 'api_contracts.db').as_posix()}"
os.environ["MODEL_ARTIFACT_DIR"] = str(Path(_TEMP_DIR.name) / "artifacts")

from api import SETTINGS, approve_model, capture_model_evaluation, capture_prediction_run, entity_profile, health, meetings, model_registry, model_status, normalize_request_id, prediction_run_detail, prediction_runs, predictions, race_card, ready, safeguards, summary
from api_contracts import (
    HealthResponse,
    ModelRegistryResponse,
    ModelSnapshotResponse,
    PredictionRunResponse,
    PredictionRunsResponse,
    PredictionsResponse,
    ProductSafeguardsResponse,
    RaceCardResponse,
    ReadinessResponse,
    SummaryResponse,
)


class APIContractTests(unittest.TestCase):
    def test_health_and_ready_responses_are_typed_and_sanitized(self) -> None:
        health_response = HealthResponse(**health())
        ready_response = ReadinessResponse(**ready())

        self.assertEqual("ok", health_response.status)
        self.assertIn(ready_response.status, {"ok", "degraded"})
        self.assertEqual("sqlite", health_response.database.engine)
        self.assertNotIn("sqlite:///", str(health_response.database.database))

    def test_predictions_support_filtering_and_pagination(self) -> None:
        response = PredictionsResponse(**predictions(track="York", limit=2))

        self.assertEqual(2, response.page.returned)
        self.assertEqual(3, response.page.total)
        self.assertTrue(all(row.track == "York" for row in response.predictions))

    def test_summary_includes_data_freshness_signal(self) -> None:
        response = SummaryResponse(**summary())

        self.assertIn(response.dataFreshness.status, {"fresh", "stale", "missing", "unknown"})
        self.assertEqual(response.lastRefresh, response.dataFreshness.lastRefresh)

    def test_safeguards_endpoint_exposes_launch_policy_contract(self) -> None:
        response = ProductSafeguardsResponse(**safeguards())

        self.assertIn("not betting advice", response.responsibleUseNotice.lower())
        self.assertGreaterEqual(len(response.limitations), 3)
        self.assertIn("responsibleGambling", response.links)
        self.assertIn("privacyPolicy", response.links)
        self.assertIn("termsOfUse", response.links)

    def test_model_registry_exposes_persisted_evaluation_snapshots(self) -> None:
        snapshot = ModelSnapshotResponse(**capture_model_evaluation(None))
        response = ModelRegistryResponse(**model_registry())

        self.assertGreaterEqual(response.page.total, 1)
        self.assertEqual(snapshot.model.id, response.models[0].id)
        self.assertTrue(snapshot.model.artifactReady)
        self.assertIsNotNone(snapshot.model.artifactSha256)
        self.assertIsNotNone(snapshot.model.featureSchemaHash)
        self.assertIn("runner_brier_score", response.models[0].metrics)

    def test_model_approval_requires_and_serves_persisted_artifact(self) -> None:
        snapshot = ModelSnapshotResponse(**capture_model_evaluation(None))
        approved = ModelSnapshotResponse(**approve_model(snapshot.model.id, None))
        status = model_status()
        run = PredictionRunResponse(**capture_prediction_run(None, require_approved_model=True))

        self.assertEqual("approved", approved.model.status)
        self.assertTrue(approved.model.artifactReady)
        self.assertEqual("artifact", status["servingMode"])
        self.assertEqual(approved.model.id, status["modelVersionId"])
        self.assertEqual(approved.model.id, run.run.modelVersionId)

    def test_prediction_runs_expose_persisted_scoring_snapshots(self) -> None:
        snapshot = PredictionRunResponse(**capture_prediction_run(None))
        list_response = PredictionRunsResponse(**prediction_runs())
        detail_response = PredictionRunResponse(**prediction_run_detail(snapshot.run.id, limit=2))

        self.assertGreaterEqual(list_response.page.total, 1)
        self.assertEqual(snapshot.run.id, list_response.runs[0].id)
        self.assertEqual(snapshot.run.runnerCount, snapshot.page.total)
        self.assertEqual(2, detail_response.page.returned)
        self.assertEqual(snapshot.run.runnerCount, detail_response.page.total)
        self.assertIsNotNone(snapshot.entries[0].win_probability)

    def test_request_id_sanitizer_rejects_unsafe_values(self) -> None:
        self.assertEqual("trace-123_:.ok", normalize_request_id("trace-123_:.ok"))

        sanitized = normalize_request_id("../" * 40)

        self.assertEqual(32, len(sanitized))
        self.assertNotIn("/", sanitized)

    def test_deployed_runtime_rejects_weak_launch_config(self) -> None:
        candidate = replace(
            SETTINGS,
            app_env="production",
            database_url="sqlite:///local.db",
            backend_cors_origins=("http://horse-predictor.example.com",),
            allowed_hosts=(),
            api_auth_token="short",
            max_request_body_bytes=512,
        )

        with self.assertRaises(RuntimeError) as context:
            candidate.validate_runtime()

        message = str(context.exception)
        self.assertIn("managed SQL", message)
        self.assertIn("ALLOWED_HOSTS", message)
        self.assertIn("HTTPS", message)
        self.assertIn("32 characters", message)
        self.assertIn("MAX_REQUEST_BODY_BYTES", message)

    def test_race_card_supports_runner_search(self) -> None:
        response = RaceCardResponse(**race_card(horse="Golden"))

        self.assertEqual(1, response.page.total)
        self.assertEqual("Golden Arrow", response.raceCard[0].horse)

    def test_meetings_and_entity_profile_endpoints_return_core_shapes(self) -> None:
        meeting_response = meetings()
        profile = entity_profile("jockey", "A. Lee")

        self.assertGreaterEqual(meeting_response["page"].total, 1)
        self.assertEqual("jockey", profile["entityType"])
        self.assertEqual(2, profile["runs"])
        self.assertEqual(1, profile["wins"])


if __name__ == "__main__":
    unittest.main()
