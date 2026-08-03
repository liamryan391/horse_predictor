from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


managed_db = load_script("managed-db-rehearsal.py")
staging_env = load_script("staging-env-check.py")


class ManagedDeploymentTests(unittest.TestCase):
    def test_managed_db_plan_redacts_passwords(self) -> None:
        args = managed_db.parse_args(
            [
                "--mode",
                "roundtrip",
                "--source-url",
                "mysql+pymysql://horse_user:secret-source@db.example.net:3306/horse_predictor",
                "--restore-url",
                "mysql+pymysql://horse_restore:secret-restore@restore.example.net:3306/horse_predictor_restore",
                "--backup-path",
                "backups/test.sql",
            ]
        )

        plan = managed_db.build_plan(args)

        self.assertEqual("planned", plan["status"])
        self.assertIn("***", plan["source"]["redactedUrl"])
        self.assertIn("***", plan["restore"]["redactedUrl"])
        self.assertNotIn("secret-source", str(plan))
        self.assertNotIn("secret-restore", str(plan))
        self.assertEqual(["backup", "restore"], [step["name"] for step in plan["steps"]])

    def test_managed_db_rejects_same_restore_target(self) -> None:
        args = managed_db.parse_args(
            [
                "--mode",
                "roundtrip",
                "--source-url",
                "mysql+pymysql://horse_user:secret@db.example.net:3306/horse_predictor",
                "--restore-url",
                "mysql+pymysql://horse_user:secret@db.example.net:3306/horse_predictor",
            ]
        )

        plan = managed_db.build_plan(args)

        self.assertEqual("failed", plan["status"])
        self.assertTrue(any("must not be the same" in error for error in plan["errors"]))

    def test_staging_env_check_passes_redacted_deployed_config(self) -> None:
        env_text = "\n".join(
            [
                "APP_ENV=staging",
                "DATABASE_URL=mysql+pymysql://horse_user:strong-password@db.example.net:3306/horse_predictor",
                "BACKEND_CORS_ORIGINS=https://horse-predictor-staging.example.net",
                "ALLOWED_HOSTS=horse-predictor-api-staging.example.net",
                "ACCOUNT_AUTH_ENABLED=true",
                "ACCOUNT_AUTH_TOKEN=account-token-value-with-more-than-32-chars",
                "JOURNAL_ACCOUNT_KEY=staging-operator",
                "VITE_API_BASE_URL=https://horse-predictor-api-staging.example.net",
                "MODEL_ARTIFACT_DIR=/app/model_artifacts",
                "REQUIRE_APPROVED_MODEL_ARTIFACT=true",
                "HORSE_API_PROVIDER=sample",
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env.staging"
            env_path.write_text(env_text + "\n", encoding="utf-8")
            args = staging_env.parse_args(["--env-file", str(env_path), "--require-release-token"])
            report = staging_env.build_report(staging_env.parse_env_file(env_path), args)

        self.assertEqual("passed", report["status"])
        self.assertIn("***", report["summary"]["DATABASE_URL"])
        self.assertNotIn("strong-password", str(report))
        self.assertIn("<set:", report["summary"]["ACCOUNT_AUTH_TOKEN"])

    def test_staging_env_check_rejects_sqlite_and_placeholders(self) -> None:
        env_text = "\n".join(
            [
                "APP_ENV=production",
                "DATABASE_URL=sqlite:///local.db",
                "BACKEND_CORS_ORIGINS=http://example.com",
                "ALLOWED_HOSTS=*",
                "ACCOUNT_AUTH_ENABLED=true",
                "JOURNAL_ACCOUNT_KEY=replace-me",
                "VITE_API_BASE_URL=http://example.com",
                "MODEL_ARTIFACT_DIR=/app/model_artifacts",
                "REQUIRE_APPROVED_MODEL_ARTIFACT=true",
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env.production"
            env_path.write_text(env_text + "\n", encoding="utf-8")
            args = staging_env.parse_args(["--env-file", str(env_path)])
            report = staging_env.build_report(staging_env.parse_env_file(env_path), args)

        self.assertEqual("failed", report["status"])
        self.assertTrue(any("managed SQL" in error for error in report["errors"]))
        self.assertTrue(any("placeholder" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
