from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("provider_smoke_check", ROOT / "scripts" / "provider-smoke-check.py")
provider_smoke_check = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(provider_smoke_check)


class ProviderSmokeCheckTests(unittest.TestCase):
    def test_redacted_config_reports_secret_shape_only(self) -> None:
        config = provider_smoke_check.APIConfig(
            provider="ourhub",
            base_url="https://api.example.com",
            api_key="secret-key-value",
            username="user@example.com",
            password="password-value",
        )

        payload = provider_smoke_check.redacted_config(config)

        self.assertEqual({"present": True, "length": len("secret-key-value")}, payload["apiKey"])
        self.assertEqual({"present": True, "length": len("user@example.com")}, payload["username"])
        self.assertEqual({"present": True, "length": len("password-value")}, payload["password"])
        self.assertNotIn("secret-key-value", str(payload))
        self.assertNotIn("password-value", str(payload))

    def test_ourhub_acceptance_allows_missing_odds_and_owner(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "track": "Windsor",
                    "distance": 1320.0,
                    "surface": "Good",
                    "horse": "North Ridge",
                    "jockey": "A. Rider",
                    "trainer": "T. Trainer",
                    "owner": None,
                    "odds": None,
                }
            ]
        )
        summary = provider_smoke_check.frame_summary(frame)

        failures = provider_smoke_check.acceptance_failures(
            "ourhub",
            {"rows": 0, "coverage": {}},
            summary,
            [],
            provider_smoke_check.DEFAULT_COVERAGE_THRESHOLDS["ourhub"],
            min_historical_rows=0,
            min_current_rows=1,
            max_error_issues=0,
            max_warning_issues=None,
        )

        self.assertEqual([], failures)

    def test_field_thresholds_fail_when_required_coverage_is_low(self) -> None:
        frame = pd.DataFrame(
            [
                {"track": "Windsor", "distance": None, "surface": "Good", "horse": "Runner"},
                {"track": "Windsor", "distance": 1320.0, "surface": "Good", "horse": "Other Runner"},
            ]
        )
        summary = provider_smoke_check.frame_summary(frame)

        failures = provider_smoke_check.acceptance_failures(
            "generic",
            {"rows": 0, "coverage": {}},
            summary,
            [],
            {"distance": 0.75},
            min_historical_rows=0,
            min_current_rows=1,
            max_error_issues=0,
            max_warning_issues=None,
        )

        self.assertEqual(["distance coverage 50% below threshold 75%"], failures)

    def test_built_in_smoke_check_ignores_env_base_url_by_default(self) -> None:
        self.assertEqual(
            "https://api.ourhub.site/api",
            provider_smoke_check.default_base_url("ourhub", "https://racing.ourhub.site/"),
        )

    def test_built_in_smoke_check_allows_explicit_base_url_override(self) -> None:
        self.assertEqual(
            "https://proxy.example.com",
            provider_smoke_check.default_base_url("ourhub", "https://proxy.example.com", allow_override=True),
        )


if __name__ == "__main__":
    unittest.main()
