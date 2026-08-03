from __future__ import annotations

from dataclasses import dataclass
import unittest

from provider_adapters import (
    APIConfig,
    fetch_paginated,
    flatten_ourhub_payload,
    flatten_racing_api_payload,
    has_http_base_url,
    normalize_provider_records,
    resolve_provider_base_url,
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

    def test_ourhub_fixture_matches_runner_label_to_course_metadata(self) -> None:
        course_payload = {
            "Market Rasen": [
                {
                    "race_time": "14:10",
                    "race_name": "Summer Handicap",
                    "distance": "2m1f",
                    "going": "Good",
                    "race_class": "Class 4",
                }
            ]
        }
        runner_payload = {
            "Market Rasen 14:10 Summer Handicap": [
                {
                    "horse_name": "North Ridge",
                    "jockey_name": "A. Rider",
                    "trainer_name": "T. Trainer",
                    "weight": "11-2",
                    "number": "3",
                }
            ]
        }

        frame, issues = flatten_ourhub_payload(course_payload, runner_payload, "2026-08-03")

        self.assertEqual(1, len(frame))
        self.assertEqual("Market Rasen", frame.loc[0, "track"])
        self.assertEqual(3740.0, frame.loc[0, "distance"])
        self.assertEqual("Good", frame.loc[0, "surface"])
        self.assertEqual("North Ridge", frame.loc[0, "horse"])
        self.assertFalse([issue for issue in issues if issue.severity == "error"])
        self.assertFalse([issue for issue in issues if issue.field in {"distance", "surface"}])

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

    def test_base_url_validation_rejects_provider_names(self) -> None:
        self.assertTrue(has_http_base_url("https://api.example.com"))
        self.assertFalse(has_http_base_url("ourhub"))
        self.assertFalse(has_http_base_url(""))

    def test_built_in_providers_use_official_base_url_by_default(self) -> None:
        self.assertEqual(
            "https://api.ourhub.site/api",
            resolve_provider_base_url("ourhub", "https://racing.ourhub.site/"),
        )
        self.assertEqual(
            "https://api.theracingapi.com",
            resolve_provider_base_url("theracingapi", "https://racing.ourhub.site/"),
        )

    def test_built_in_provider_base_url_can_be_explicitly_overridden(self) -> None:
        self.assertEqual(
            "https://proxy.example.com",
            resolve_provider_base_url("ourhub", "https://proxy.example.com", allow_override=True),
        )


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
