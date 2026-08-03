from __future__ import annotations

import unittest

from local_ai import (
    LocalAIConfig,
    chat_completions_url,
    chat_request_payload,
    review_payload_mapping,
    validate_mapping_review,
)


class LocalAITests(unittest.TestCase):
    def test_disabled_ai_review_is_non_network_status(self) -> None:
        response = review_payload_mapping({"type": "object"}, LocalAIConfig())

        self.assertEqual("disabled", response["status"])
        self.assertFalse(response["config"]["enabled"])

    def test_chat_request_uses_json_object_response_format(self) -> None:
        payload = chat_request_payload(
            LocalAIConfig(provider="ollama", model="llama3.1"),
            [{"role": "user", "content": "Return JSON."}],
        )

        self.assertEqual("llama3.1", payload["model"])
        self.assertEqual({"type": "json_object"}, payload["response_format"])
        self.assertEqual("http://127.0.0.1:11434/v1/chat/completions", chat_completions_url("http://127.0.0.1:11434/v1"))

    def test_mapping_review_validation_requires_strict_lists(self) -> None:
        valid = {
            "status": "needs_review",
            "notes": ["runner keys are present"],
            "suggestedAliases": ["York Racecourse -> York"],
            "fieldMappings": ["horse_name -> horse"],
            "warnings": ["odds missing"],
        }

        self.assertEqual(valid, validate_mapping_review(valid))

        invalid = {**valid, "warnings": "odds missing"}
        with self.assertRaises(ValueError):
            validate_mapping_review(invalid)


if __name__ == "__main__":
    unittest.main()
