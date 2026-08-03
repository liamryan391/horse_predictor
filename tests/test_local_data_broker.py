from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_data_broker import (
    list_cached_payloads,
    load_cached_payload,
    manual_json_to_provider_result,
    payload_shape,
    raw_payload_cache_status,
    write_raw_payload,
)


class LocalDataBrokerTests(unittest.TestCase):
    def test_raw_payload_cache_writes_redigestible_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = {"results": [{"horse": "Golden Arrow", "track": "York"}]}

            record = write_raw_payload(
                temp_dir,
                "ourhub",
                "runner_info",
                payload,
                endpoint="/runner-info/2026-08-03",
                params={"day": "today"},
                source_url="https://api.example.com/runner-info/2026-08-03?day=today",
                license_reference="test license",
                fetched_at="2026-08-03T12:00:00+00:00",
            )

            self.assertTrue(Path(record.path).exists())
            metadata, loaded_payload = load_cached_payload(record.path)
            self.assertEqual(payload, loaded_payload)
            self.assertEqual(record.payload_sha256, metadata["payloadSha256"])
            self.assertEqual("test license", metadata["licenseReference"])

            cached = list_cached_payloads(temp_dir)
            self.assertEqual(1, len(cached))
            self.assertEqual("ourhub", cached[0].provider)
            self.assertEqual(1, raw_payload_cache_status(temp_dir)["rawPayloadCount"])

    def test_manual_json_to_provider_result_uses_existing_race_schema(self) -> None:
        payload = {
            "historical": [
                {
                    "race_date": "2026-08-03",
                    "track": "York",
                    "distance": 1760,
                    "surface": "Good",
                    "horse": "Broker Winner",
                    "jockey": "A. Local",
                    "owner": "Local Owner",
                    "trainer": "T. Broker",
                    "odds": 4.2,
                    "finishing_position": 1,
                }
            ],
            "current": [
                {
                    "race_date": "2026-08-03",
                    "track": "York",
                    "distance": 1760,
                    "surface": "Good",
                    "horse": "Broker Runner",
                    "jockey": "A. Local",
                    "owner": "Local Owner",
                    "trainer": "T. Broker",
                    "odds": 5.5,
                }
            ],
        }

        result = manual_json_to_provider_result(payload)

        self.assertEqual(1, len(result.historical))
        self.assertEqual(1, len(result.current))
        self.assertEqual("Broker Runner", result.current.loc[0, "horse"])
        self.assertFalse([issue for issue in result.validation_issues if issue.severity == "error"])
        self.assertEqual(1, len(result.raw_payloads))

    def test_payload_shape_summarizes_nested_lists_without_values(self) -> None:
        shape = payload_shape({"runners": [{"horse": "A", "odds": 3.0}], "course": "York"})

        self.assertEqual("object", shape["type"])
        self.assertEqual({"rows": 1, "sampleKeys": ["horse", "odds"]}, shape["listFields"]["runners"])


if __name__ == "__main__":
    unittest.main()
