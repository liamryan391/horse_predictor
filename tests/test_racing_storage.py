from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest

from alembic import command
from alembic.config import Config
import pandas as pd
from sqlalchemy import create_engine, inspect, text

from prediction_model import build_feature_table, evaluate_model, save_model_artifact, score_current_races, train_model
from racing_storage import (
    TABLES,
    approve_model_version,
    create_or_update_operator_account,
    delete_bet_journal_entry,
    ingestion_status,
    normalize_account_roles,
    read_latest_approved_model_version,
    read_admin_audit_events,
    read_bet_journal,
    read_model_registry,
    read_normalized_race_entries,
    read_operator_account_by_token,
    read_operator_accounts,
    read_prediction_run,
    read_prediction_runs,
    read_races,
    normalized_table_counts,
    provider_freshness_report,
    record_bet_journal_entry,
    record_admin_audit_event,
    record_model_evaluation_snapshot,
    record_prediction_run,
    release_job_lock,
    sync_normalized_from_compatibility,
    table_counts,
    try_acquire_job_lock,
    update_model_version_status,
    update_bet_journal_entry,
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

    def test_normalized_sync_populates_provider_entity_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "normalized.db")
            historical = pd.DataFrame(
                [
                    {
                        "race_date": "2026-08-01",
                        "track": "York",
                        "distance": 1760,
                        "surface": "Good",
                        "horse": "Stable One",
                        "jockey": "A. Local",
                        "owner": "Local Owner",
                        "trainer": "T. Trainer",
                        "odds": 4.5,
                        "finishing_position": 1,
                        "draw": 3,
                    }
                ]
            )
            current = pd.DataFrame(
                [
                    {
                        "race_date": "2026-08-03",
                        "track": "York",
                        "distance": 1760,
                        "surface": "Good",
                        "horse": "Stable Two",
                        "jockey": "B. Local",
                        "owner": "Local Owner",
                        "trainer": "T. Trainer",
                        "odds": 5.0,
                        "draw": 4,
                    }
                ]
            )

            write_races(database_url, TABLES["historical"], historical, source="phase21", replace=True)
            write_races(database_url, TABLES["current"], current, source="phase21", replace=True)
            sync_counts = sync_normalized_from_compatibility(database_url, source="phase21")
            counts = normalized_table_counts(database_url)
            entries = read_normalized_race_entries(database_url, provider="phase21", track="York")

            self.assertEqual(2, sync_counts["raceEntries"])
            self.assertEqual(1, counts["courses"])
            self.assertEqual(2, counts["races"])
            self.assertEqual(2, counts["race_entries"])
            self.assertEqual(1, counts["historical_results"])
            self.assertEqual(2, counts["odds_snapshots"])
            self.assertEqual(2, entries["page"]["total"])
            self.assertTrue(all(entry["providerEntryId"].startswith("synthetic:") for entry in entries["entries"]))
            self.assertEqual(1.0, entries["entries"][0]["finishingPosition"])

    def test_job_lock_prevents_duplicate_ingestion_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "locks.db")

            self.assertTrue(try_acquire_job_lock(database_url, "ingestion-worker", "owner-a", 60))
            self.assertFalse(try_acquire_job_lock(database_url, "ingestion-worker", "owner-b", 60))
            self.assertFalse(release_job_lock(database_url, "ingestion-worker", "owner-b"))
            self.assertTrue(release_job_lock(database_url, "ingestion-worker", "owner-a"))
            self.assertTrue(try_acquire_job_lock(database_url, "ingestion-worker", "owner-b", 60))

    def test_operator_accounts_hash_tokens_and_track_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "accounts.db")
            token = "phase22-storage-token-12345"

            self.assertEqual(["viewer", "journal"], normalize_account_roles(["journal"]))

            created = create_or_update_operator_account(
                database_url,
                account_key="liam",
                display_name="Liam",
                email="liam@example.com",
                roles=["journal"],
                token=token,
                privacy_acknowledged=True,
            )
            matched = read_operator_account_by_token(database_url, token)
            missing = read_operator_account_by_token(database_url, "wrong-token-value-12345")
            listed = read_operator_accounts(database_url)

            self.assertEqual("liam", created["accountKey"])
            self.assertEqual(["viewer", "journal"], created["roles"])
            self.assertTrue(created["tokenConfigured"])
            self.assertIsNotNone(created["privacyAcknowledgedAt"])
            self.assertNotIn("tokenSha256", created)
            self.assertIsNotNone(matched)
            self.assertEqual(created["id"], matched["id"])
            self.assertEqual(64, len(matched["tokenSha256"]))
            self.assertIsNotNone(matched["lastAuthenticatedAt"])
            self.assertIsNone(missing)
            self.assertEqual(1, listed["page"]["total"])
            self.assertNotIn("tokenSha256", listed["accounts"][0])

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

            superseded = update_model_version_status(database_url, snapshot["id"], "superseded")

            self.assertIsNotNone(superseded)
            self.assertEqual("superseded", superseded["status"])

    def test_admin_audit_events_record_actor_roles_and_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "audit.db")

            event = record_admin_audit_event(
                database_url,
                actor="liam",
                roles=("reader", "admin"),
                action="model_version.approve",
                resource_type="model_version",
                resource_id=12,
                request_id="req-123",
                detail="approved from unit test",
                payload={"artifactReady": True},
            )
            events = read_admin_audit_events(database_url)

            self.assertEqual("liam", event["actor"])
            self.assertEqual(["reader", "admin"], event["roles"])
            self.assertEqual("model_version", event["resourceType"])
            self.assertEqual("12", event["resourceId"])
            self.assertEqual({"artifactReady": True}, event["payload"])
            self.assertEqual(1, events["page"]["total"])
            self.assertEqual(event["id"], events["events"][0]["id"])

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

    def test_bet_journal_crud_uses_server_side_user_bets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "bets.db")

            bet = record_bet_journal_entry(
                database_url,
                {
                    "horse": "Golden Arrow",
                    "track": "York",
                    "raceDate": "2026-08-03",
                    "stake": 10,
                    "odds": 3.5,
                    "notes": "phase 15 smoke",
                },
            )

            self.assertEqual("Golden Arrow", bet["horse"])
            self.assertEqual("open", bet["status"])
            self.assertIsNone(bet["profitLoss"])

            journal = read_bet_journal(database_url)

            self.assertEqual(1, journal["page"]["total"])
            self.assertEqual(bet["id"], journal["bets"][0]["id"])

            updated = update_bet_journal_entry(
                database_url,
                bet["id"],
                {"status": "won", "closingOdds": 3.1, "notes": "settled"},
            )

            self.assertIsNotNone(updated)
            self.assertEqual("won", updated["status"])
            self.assertAlmostEqual(25.0, updated["profitLoss"])
            self.assertEqual(3.1, updated["closingOdds"])
            self.assertIsNotNone(updated["settledAt"])

            won_only = read_bet_journal(database_url, status="won")

            self.assertEqual(1, won_only["page"]["total"])
            self.assertTrue(delete_bet_journal_entry(database_url, bet["id"]))
            self.assertEqual(0, read_bet_journal(database_url)["page"]["total"])

    def test_legacy_sqlite_user_bets_table_is_repaired_for_server_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = _sqlite_url(Path(temp_dir) / "legacy-bets.db")
            engine = create_engine(database_url)
            try:
                with engine.begin() as conn:
                    conn.execute(text("CREATE TABLE race_entries (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT)"))
                    conn.execute(
                        text(
                            """
                            CREATE TABLE user_bets (
                                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                                race_entry_id INTEGER NOT NULL,
                                bet_type VARCHAR(80) NOT NULL,
                                stake FLOAT NOT NULL,
                                odds_decimal FLOAT,
                                status VARCHAR(80) NOT NULL,
                                placed_at DATETIME NOT NULL,
                                settled_at DATETIME,
                                profit_loss FLOAT,
                                notes TEXT
                            )
                            """
                        )
                    )
            finally:
                engine.dispose()

            bet = record_bet_journal_entry(
                database_url,
                {"horse": "Legacy Runner", "stake": 5, "odds": 2.8},
                account_key="legacy-account",
            )
            journal = read_bet_journal(database_url, account_key="legacy-account")
            engine = create_engine(database_url)
            try:
                with engine.connect() as conn:
                    columns = {column["name"]: column for column in inspect(conn).get_columns("user_bets")}
            finally:
                engine.dispose()

            self.assertEqual("Legacy Runner", bet["horse"])
            self.assertEqual("legacy-account", bet["accountKey"])
            self.assertEqual(1, journal["page"]["total"])
            self.assertIn("account_key", columns)
            self.assertTrue(columns["race_entry_id"]["nullable"])

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
                        bet_columns = {column["name"]: column for column in inspector.get_columns("user_bets")}
                finally:
                    engine.dispose()
                self.assertIn("races_current", table_names)
                self.assertIn("admin_audit_events", table_names)
                self.assertIn("operator_accounts", table_names)
                self.assertIn("model_evaluation_results", table_names)
                self.assertIn("prediction_run_entries", table_names)
                self.assertIn("account_key", bet_columns)
                self.assertIn("horse", bet_columns)
                self.assertIn("closing_odds_decimal", bet_columns)
                self.assertTrue(bet_columns["race_entry_id"]["nullable"])
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
