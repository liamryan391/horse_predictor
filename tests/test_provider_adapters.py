from __future__ import annotations

from dataclasses import dataclass
import unittest

from provider_adapters import (
    APIConfig,
    fetch_paginated,
    flatten_racing_api_payload,
    normalize_provider_records,
    summarize_validation_issues,
)


class ProviderAdapterTests(unittest.TestCase):
    def test_racing_api_fixture_flattens_nested_runner_payload(self) -> None:
        payload = {
            "racecards": [
                {
                    "date": "2026-05-05",
                    "course": "York",
                    "distance": "1m2f",
                    "going": "Good",
                    "runners": [
                        {
                            "horse": "Golden Arrow",
                            "jockey": "C. Smith",
                            "trainer": "P. Miller",
                            "owner": "Green Acres",
                            "odds": {"decimal": 3.4},
                            "position": 1,
                        }
                    ],
                }
            ]
        }

        frame, issues = flatten_racing_api_payload(payload)

        self.assertEqual(1, len(frame))
        self.assertEqual("Golden Arrow", frame.loc[0, "horse"])
        self.assertEqual(2200.0, frame.loc[0, "distance"])
        self.assertEqual(3.4, frame.loc[0, "odds"])
        self.assertEqual("GB", frame.loc[0, "country"])
        self.assertEqual("middle", frame.loc[0, "distance_bucket"])
        self.assertEqual("good", frame.loc[0, "going_category"])
        self.assertFalse([issue for issue in issues if issue.severity == "error"])

    def test_provider_validation_summarizes_missing_required_fields(self) -> None:
        frame, issues = normalize_provider_records(
            [{"race_date": "", "track": "York", "horse": "", "finishing_position": None}],
            current_mode=False,
        )

        self.assertTrue(frame.empty)
        self.assertEqual("race_date", issues[0].field)
        self.assertIn("3 errors", summarize_validation_issues(issues) or "")

    def test_paginated_fetch_stops_at_configured_page_limit(self) -> None:
        client = FakeClient(max_pages=2)

        records = fetch_paginated(client, "/results", {"limit": 50}, ["results"])

        self.assertEqual([1, 2], [record["page"] for record in records])
        self.assertEqual([1, 2], client.pages_requested)


@dataclass
class FakeConfig:
    max_pages: int


class FakeClient:
    def __init__(self, max_pages: int) -> None:
        self.config = FakeConfig(max_pages=max_pages)
        self.pages_requested: list[int] = []

    def fetch_json(self, endpoint: str, params: dict | None = None) -> dict:
        _ = endpoint
        page = int((params or {})["page"])
        self.pages_requested.append(page)
        return {"results": [{"page": page}], "pagination": {"next_page": page + 1}}


if __name__ == "__main__":
    unittest.main()
