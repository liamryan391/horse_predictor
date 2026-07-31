from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest

from alembic import command
from alembic.config import Config
import pandas as pd
from sqlalchemy import create_engine, inspect

from racing_storage import TABLES, ingestion_status, read_races, table_counts, write_races
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
            self.assertEqual(1, len(stored))
            self.assertEqual(9.9, stored.loc[0, "odds"])
            self.assertEqual(1, table_counts(database_url)["current"])
            self.assertEqual("success", ingestion_status(database_url).loc[0, "status"])

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
                    table_names = set(inspect(engine).get_table_names())
                finally:
                    engine.dispose()
                self.assertIn("races_current", table_names)
                self.assertIn("model_evaluation_results", table_names)

                command.downgrade(config, "base")
                engine = create_engine(database_url)
                try:
                    table_names = set(inspect(engine).get_table_names())
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
