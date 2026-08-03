from __future__ import annotations

from urllib.parse import parse_qs, urlparse
import unittest

import pandas as pd

from race_enrichment import (
    course_metadata_for,
    distance_bucket,
    enrich_race_frame,
    going_category,
    infer_race_type,
    open_meteo_daily_weather_url,
)


class RaceEnrichmentTests(unittest.TestCase):
    def test_course_metadata_matches_normalized_course_names(self) -> None:
        metadata = course_metadata_for("York Racecourse")

        self.assertIsNotNone(metadata)
        self.assertEqual("York", metadata.name)
        self.assertEqual("GB", metadata.country)

    def test_distance_going_and_race_type_buckets_are_stable(self) -> None:
        self.assertEqual("sprint", distance_bucket(1200))
        self.assertEqual("mile", distance_bucket(1760))
        self.assertEqual("middle", distance_bucket("1m2f"))
        self.assertEqual("staying", distance_bucket(3300))

        self.assertEqual("soft", going_category("Turf", "Soft"))
        self.assertEqual("all_weather", going_category("Dirt", None))
        self.assertEqual("jumps", infer_race_type("Hurdle", None))
        self.assertEqual("flat_aw", infer_race_type("Dirt", None))

    def test_enrich_race_frame_adds_course_and_model_context(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "track": "York",
                    "distance": 1400,
                    "surface": "Dirt",
                    "weather": "Sunny",
                }
            ]
        )

        enriched = enrich_race_frame(frame)

        self.assertEqual("GB", enriched.loc[0, "country"])
        self.assertEqual("sprint", enriched.loc[0, "distance_bucket"])
        self.assertEqual("all_weather", enriched.loc[0, "going_category"])
        self.assertEqual("flat_aw", enriched.loc[0, "race_type"])
        self.assertAlmostEqual(53.9399, enriched.loc[0, "course_latitude"], places=4)

    def test_open_meteo_url_targets_archive_daily_weather(self) -> None:
        url = open_meteo_daily_weather_url(53.9399, -1.0973, "2025-03-18")
        parsed = urlparse(url)
        params = parse_qs(parsed.query)

        self.assertEqual("archive-api.open-meteo.com", parsed.netloc)
        self.assertEqual(["2025-03-18"], params["start_date"])
        self.assertEqual(["2025-03-18"], params["end_date"])
        self.assertIn("temperature_2m_mean", params["daily"][0])
        self.assertIn("precipitation_sum", params["daily"][0])


if __name__ == "__main__":
    unittest.main()
