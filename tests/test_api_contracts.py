from __future__ import annotations

import unittest

from api import entity_profile, health, meetings, predictions, race_card, ready
from api_contracts import HealthResponse, PredictionsResponse, RaceCardResponse, ReadinessResponse


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
