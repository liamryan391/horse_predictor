# Local AI And DIY Data API Strategy

This note captures the cost-aware direction after the August 3, 2026 live provider check.

## Short Answer

Building a local API is a good later-stage idea, but it should be a data broker, not a replacement for real race data.

Use local AI for:

- parsing and normalizing allowed provider payloads
- suggesting track aliases and field mappings
- explaining model outputs in plain language
- flagging missing fields and suspicious provider changes
- generating local-only race summaries with clear caveats

Do not use local AI as the source of truth for:

- today's runners
- race times
- official results
- live odds
- settled positions
- stable race or runner identifiers

Those still need licensed providers, official sources, or manually imported data with provenance.

## What "Local OpenAI API" Means

There is no free local version of OpenAI's hosted API. OpenAI's official API is cloud-hosted and usage-priced.

There are local model servers that provide OpenAI-compatible endpoints. They let the app use the familiar OpenAI client shape while running open-weight models on this machine:

- Ollama: OpenAI-compatible endpoints for connecting existing apps to local Ollama models.
- LM Studio: local server with OpenAI-compatible and Anthropic-compatible endpoints.
- llama-cpp-python: OpenAI-compatible web server for local GGUF models.

These are free in API-billing terms, but they use local CPU/GPU, memory, disk, and electricity. Output quality depends on the model and hardware.

## Recommended Architecture

Add a local data broker service inside this project. Phase 20 starts this in-process with `local_data_broker.py`, opt-in raw payload cache envelopes, manual JSON normalization, and `/api/v1/broker/...` visibility endpoints:

```text
provider feeds / files
        |
        v
local data broker API
        |
        +-- raw payload cache
        +-- deterministic field mappers
        +-- provider provenance and licensing notes
        +-- optional local LLM parser/reviewer
        |
        v
Horse Predictor DB and API
```

The current broker exposes:

- `/api/v1/broker/status`
- `/api/v1/broker/raw-payloads`
- `/api/v1/broker/raw-payload-shape`

Future broker endpoints can now build on the Phase 21 normalized provider entity model to add app-owned racecards, results, odds, provider plugin status, and import job controls.

The current `data_pipeline.py` provider adapters can then call the broker instead of calling every external provider directly.

## Local AI Provider Settings

Phase 20 adds settings like:

```text
AI_PROVIDER=disabled
AI_BASE_URL=http://127.0.0.1:11434/v1
AI_MODEL=
AI_TIMEOUT_SECONDS=20
AI_JSON_SCHEMA_REQUIRED=true
AI_API_KEY=
```

Suggested provider values:

- `disabled`
- `ollama`
- `lmstudio`
- `llama_cpp`
- `openai-compatible`

Start with `disabled` as the default. Local AI should be opt-in and should never block core ingestion.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\local-ai-smoke-check.py --provider ollama --base-url http://127.0.0.1:11434/v1 --model llama3.1
```

See [LOCAL_DATA_BROKER.md](LOCAL_DATA_BROKER.md) for broker cache and local AI commands.

## Guardrails

- Validate every AI output with a strict JSON schema.
- Store the original provider payload beside any AI-normalized result.
- Mark AI-derived fields separately from provider-supplied fields.
- Require human review before adopting new mappings.
- Never infer missing official results or odds.
- Do not scrape or redistribute data unless the source terms permit it.
- Keep model predictions deterministic and auditable; use LLM output for context, not winner selection.

## Provider Cost Implication

The Racing API paid tiers may be useful later, but they should not be a mandatory dependency while the project is still proving product value.

Near-term priority should be:

1. Use OurHub for live race-card display and track matching.
2. Add a local broker so imports, caches, manual files, and future providers share one app-owned contract.
3. Add a cheap or free historical-results path.
4. Add an odds-capable source only after usage rights are clear.
5. Add local AI as a helper for normalization and operator explanations.

## Source References

- Ollama OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility
- Ollama API introduction: https://docs.ollama.com/api/introduction
- LM Studio OpenAI compatibility: https://lmstudio.ai/docs/developer/openai-compat
- LM Studio local server: https://lmstudio.ai/docs/developer/core/server
- llama-cpp-python OpenAI-compatible server: https://llama-cpp-python.readthedocs.io/en/latest/server/
- OpenAI API pricing: https://developers.openai.com/api/docs/pricing
