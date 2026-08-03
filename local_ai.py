from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

DISABLED_AI_PROVIDERS = {"", "disabled", "off", "none"}
LOCAL_AI_PROVIDERS = {"ollama", "lmstudio", "llama-cpp", "llama_cpp", "openai-compatible", "local"}
REQUIRED_REVIEW_KEYS = {"status", "notes", "suggestedAliases", "fieldMappings", "warnings"}


@dataclass(frozen=True)
class LocalAIConfig:
    provider: str = "disabled"
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = ""
    timeout_seconds: float = 20.0
    json_schema_required: bool = True
    api_key: str = ""

    @property
    def enabled(self) -> bool:
        return self.provider.strip().lower() not in DISABLED_AI_PROVIDERS


def local_ai_config_from_settings(settings: Any) -> LocalAIConfig:
    return LocalAIConfig(
        provider=settings.ai_provider,
        base_url=settings.ai_base_url,
        model=settings.ai_model,
        timeout_seconds=settings.ai_timeout_seconds,
        json_schema_required=settings.ai_json_schema_required,
        api_key=settings.ai_api_key,
    )


def redacted_ai_config(config: LocalAIConfig) -> dict[str, Any]:
    return {
        "provider": config.provider,
        "enabled": config.enabled,
        "baseUrl": config.base_url,
        "model": config.model,
        "timeoutSeconds": config.timeout_seconds,
        "jsonSchemaRequired": config.json_schema_required,
        "apiKey": {"present": bool(config.api_key.strip()), "length": len(config.api_key.strip())},
    }


def chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def mapping_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(REQUIRED_REVIEW_KEYS),
        "properties": {
            "status": {"type": "string", "enum": ["ok", "needs_review", "blocked"]},
            "notes": {"type": "array", "items": {"type": "string"}},
            "suggestedAliases": {"type": "array", "items": {"type": "string"}},
            "fieldMappings": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }


def build_mapping_review_messages(payload_summary: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You review horse racing provider payload shapes. Return JSON only. "
                "Do not invent race facts, runners, odds, or results."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Review this provider payload shape for import mapping risks.",
                    "payloadSummary": payload_summary,
                    "requiredOutput": mapping_review_schema(),
                },
                sort_keys=True,
            ),
        },
    ]


def chat_request_payload(config: LocalAIConfig, messages: list[dict[str, str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": 0,
    }
    if config.json_schema_required:
        payload["response_format"] = {"type": "json_object"}
    return payload


def extract_message_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("AI response did not include choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("AI response did not include message content.")
    return content.strip()


def parse_json_content(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("AI response was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("AI response JSON must be an object.")
    return parsed


def validate_mapping_review(payload: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_REVIEW_KEYS - set(payload)
    if missing:
        raise ValueError(f"AI mapping review is missing keys: {', '.join(sorted(missing))}.")
    status = payload.get("status")
    if status not in {"ok", "needs_review", "blocked"}:
        raise ValueError("AI mapping review status must be ok, needs_review, or blocked.")
    for key in ["notes", "suggestedAliases", "fieldMappings", "warnings"]:
        if not isinstance(payload.get(key), list) or not all(isinstance(item, str) for item in payload[key]):
            raise ValueError(f"AI mapping review {key} must be a list of strings.")
    return payload


def disabled_ai_status(config: LocalAIConfig) -> dict[str, Any]:
    return {
        "status": "disabled",
        "config": redacted_ai_config(config),
        "message": "Local AI review is disabled. Set AI_PROVIDER, AI_BASE_URL, and AI_MODEL to enable it.",
    }


def request_chat_completion(config: LocalAIConfig, messages: list[dict[str, str]]) -> dict[str, Any]:
    if not config.enabled:
        raise ValueError("Local AI is disabled.")
    if not config.model.strip():
        raise ValueError("AI_MODEL is required when local AI is enabled.")
    headers = {"Content-Type": "application/json"}
    if config.api_key.strip():
        headers["Authorization"] = f"Bearer {config.api_key.strip()}"
    response = requests.post(
        chat_completions_url(config.base_url),
        headers=headers,
        json=chat_request_payload(config, messages),
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def review_payload_mapping(payload_summary: dict[str, Any], config: LocalAIConfig) -> dict[str, Any]:
    if not config.enabled:
        return disabled_ai_status(config)
    response_payload = request_chat_completion(config, build_mapping_review_messages(payload_summary))
    content = extract_message_content(response_payload)
    review = validate_mapping_review(parse_json_content(content))
    return {"status": "success", "config": redacted_ai_config(config), "review": review}
