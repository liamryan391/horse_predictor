from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest

from alembic import command
from alembic.config import Config
import pandas as pd
from sqlalchemy import create_engine, inspect

from prediction_model import build_feature_table, evaluate_model, save_model_artifact, score_current_races, train_model
from racing_storage import (
    TABLES,
    approve_model_version,
    ingestion_status,
    read_latest_approved_model_version,
    read_model_registry,
    read_prediction_run,
    read_prediction_runs,
    read_races,
    provider_freshness_report,
    record_model_evaluation_snapshot,
    record_prediction_run,
    release_job_lock,
    table_counts,
    try_acquire_job_lock,
    write_races,
)
from settings import get_settings


ROOT = Path(__file__).resolve().parents[1]


class RacingStorageTests(unittest.TestCase):
    def test_write_races_upserts_runner_identity_instead_of_duplicating(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "storage.db")
            current = pd.read_csv(ROOT / "sample_current_races.csv").head(1)

            self.assertEqual(1, write_races(database_url, TABLES["current"], current, source="test", replace=True))
            changed = current.copy()
            changed.loc[0, "odds"] = 9.9
            self.assertEqual(1, write_races(database_url, TABLES["current"], changed, source="test"))

            stored = read_races(database_url, TABLES["current"])
            freshness = provider_freshness_report(database_url)

            self.assertEqual(1, len(stored))
            self.assertEqual(9.9, stored.loc[0, "odds"])
            self.assertEqual(1, table_counts(database_url)["current"])
            self.assertEqual("success", ingestion_status(database_url).loc[0, "status"])
            self.assertEqual("success", freshness[0]["status"])
            self.assertEqual(TABLES["current"], freshness[0]["tableName"])

    def test_job_lock_prevents_duplicate_ingestion_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "locks.db")

            self.assertTrue(try_acquire_job_lock(database_url, "ingestion-worker", "owner-a", 60))
            self.assertFalse(try_acquire_job_lock(database_url, "ingestion-worker", "owner-b", 60))
            self.assertFalse(release_job_lock(database_url, "ingestion-worker", "owner-b"))
            self.assertTrue(release_job_lock(database_url, "ingestion-worker", "owner-a"))
            self.assertTrue(try_acquire_job_lock(database_url, "ingestion-worker", "owner-b", 60))

    def test_model_registry_records_and_approves_evaluation_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "models.db")
            features = build_feature_table(pd.read_csv(ROOT / "sample_historical_data.csv"))
            model = train_model(features)
            evaluation = evaluate_model(features)
            artifact = save_model_artifact(model, temp_dir, code_commit_sha="storage-test")

            snapshot = record_model_evaluation_snapshot(
                database_url,
                model,
                evaluation,
                name="candidate-smoke",
                artifact_uri=artifact.uri,
                artifact_sha256=artifact.sha256,
                feature_schema_hash=artifact.feature_schema_hash,
                code_commit_sha=artifact.code_commit_sha,
            )

            self.assertEqual("candidate-smoke", snapshot["name"])
            self.assertEqual("candidate", snapshot["status"])
            self.assertTrue(snapshot["artifactReady"])
            self.assertEqual(artifact.sha256, snapshot["artifactSha256"])
            self.assertEqual("storage-test", snapshot["codeCommitSha"])
            self.assertIn("runner_brier_score", snapshot["metrics"])

            registry = read_model_registry(database_url)
            self.assertEqual(1, registry["page"]["total"])
            self.assertEqual(snapshot["id"], registry["models"][0]["id"])

            approved = approve_model_version(database_url, snapshot["id"])

            self.assertIsNotNone(approved)
            self.assertEqual("approved", approved["status"])

    def test_prediction_runs_persist_scored_runner_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "prediction-runs.db")
            features = build_feature_table(pd.read_csv(ROOT / "sample_historical_data.csv"))
            current = pd.read_csv(ROOT / "sample_current_races.csv")
            model = train_model(features)
            evaluation = evaluate_model(features)
            snapshot = record_model_evaluation_snapshot(database_url, model, evaluation)
            approved = approve_model_version(database_url, snapshot["id"])
            scored = score_current_races(model, current)

            latest = read_latest_approved_model_version(database_url)
            run = record_prediction_run(
                database_url,
                scored,
                source="test",
                model_version_id=latest["id"] if latest else None,
                notes="unit-test snapshot",
            )

            self.assertIsNotNone(approved)
            self.assertEqual(approved["id"], run["run"]["modelVersionId"])
            self.assertEqual(len(scored), run["page"]["total"])
            self.assertEqual(len(scored), run["run"]["runnerCount"])
            self.assertTrue(run["run"]["topRunner"])
            self.assertIn("win_probability", run["entries"][0])

            runs = read_prediction_runs(database_url)
            self.assertEqual(1, runs["page"]["total"])
            self.assertEqual(run["run"]["id"], runs["runs"][0]["id"])

            detail = read_prediction_run(database_url, run["run"]["id"], limit=2)

            self.assertIsNotNone(detail)
            self.assertEqual(2, detail["page"]["returned"])
            self.assertEqual(len(scored), detail["page"]["total"])

    def test_alembic_upgrade_and_downgrade_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "migration.db")
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            old_database_url = os.environ.get("DATABASE_URL")
            os.environ["DATABASE_URL"] = database_url
            get_settings.cache_clear()

            try:
                command.upgrade(config, "head")
                engine = create_engine(database_url)
                try:
                    with engine.connect() as conn:
                        inspector = inspect(conn)
                        table_names = set(inspector.get_table_names())
                        columns = {column["name"] for column in inspector.get_columns("model_versions")}
                finally:
                    engine.dispose()
                self.assertIn("races_current", table_names)
                self.assertIn("model_evaluation_results", table_names)
                self.assertIn("prediction_run_entries", table_names)
                self.assertIn("artifact_sha256", columns)
                self.assertIn("feature_schema_hash", columns)
                self.assertIn("code_commit_sha", columns)

                command.downgrade(config, "base")
                engine = create_engine(database_url)
                try:
                    with engine.connect() as conn:
                        table_names = set(inspect(conn).get_table_names())
                finally:
                    engine.dispose()
                self.assertNotIn("races_current", table_names)
            finally:
                if old_database_url is None:
                    os.environ.pop("DATABASE_URL", None)
                else:
                    os.environ["DATABASE_URL"] = old_database_url
                get_settings.cache_clear()


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


if __name__ == "__main__":
    unittest.main()
