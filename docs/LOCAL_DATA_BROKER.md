# Local Data Broker And Local AI

Phase 20 adds an app-owned broker layer so provider payloads, manual files, cache replay, and optional local AI review can share one local contract before data reaches the main database.

## What It Does

- Caches raw provider payloads as JSON envelopes when explicitly requested.
- Stores provider, resource, endpoint, source URL, fetched timestamp, row count, payload hash, and licensing note with each cached payload.
- Replays cached payload metadata and payload shape for debugging provider mapping changes.
- Normalizes manual JSON payloads through the same race schema validator used by provider adapters.
- Exposes broker cache and local AI status through API endpoints.
- Supports local OpenAI-compatible model servers for advisory payload review.

It does not make up official runners, results, odds, or race times. Those still need licensed provider data or reviewed manual imports.

## Settings

```text
BROKER_RAW_CACHE_DIR=broker_payloads
AI_PROVIDER=disabled
AI_BASE_URL=http://127.0.0.1:11434/v1
AI_MODEL=
AI_TIMEOUT_SECONDS=20
AI_JSON_SCHEMA_REQUIRED=true
AI_API_KEY=
```

Keep `AI_PROVIDER=disabled` until Ollama, LM Studio, or llama-cpp-python is running locally and the model has passed the smoke check.

## Broker Smoke Check

Run the no-network broker smoke check:

```powershell
.\.venv\Scripts\python.exe scripts\broker-smoke-check.py --cache-dir .codex_tmp\broker-smoke
```

This writes one local fixture envelope, normalizes one historical row and one current row, and reports cache status.

## Cache Live Provider Payloads

Raw provider caching is opt-in:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider ourhub --days-ahead 0 --no-csv --disable-lock --cache-raw-payloads
```

Use a throwaway database first when live-testing:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider ourhub --database-url .codex_tmp\live-broker.db --days-ahead 0 --no-csv --disable-lock --cache-raw-payloads --broker-cache-dir .codex_tmp\broker-payloads
```

## Broker API Endpoints

With the FastAPI app running:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/broker/status
Invoke-RestMethod http://127.0.0.1:8000/api/v1/broker/raw-payloads
```

Inspect one cached payload shape by passing a path from `/broker/raw-payloads`:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/broker/raw-payload-shape?path=<payload-path>"
```

Add `ai_review=true` only after local AI is configured:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/broker/raw-payload-shape?path=<payload-path>&ai_review=true"
```

The shape endpoint is bounded to `BROKER_RAW_CACHE_DIR`; it cannot read arbitrary files.

## Local AI Smoke Check

Example for Ollama:

```powershell
$env:AI_PROVIDER="ollama"
$env:AI_BASE_URL="http://127.0.0.1:11434/v1"
$env:AI_MODEL="llama3.1"
.\.venv\Scripts\python.exe scripts\local-ai-smoke-check.py
```

Example for LM Studio:

```powershell
$env:AI_PROVIDER="lmstudio"
$env:AI_BASE_URL="http://127.0.0.1:1234/v1"
$env:AI_MODEL="local-model"
.\.venv\Scripts\python.exe scripts\local-ai-smoke-check.py
```

The smoke check asks the local model to return strict JSON for a provider-payload review. It does not send provider credentials.

## Guardrails

- Cache raw payloads only when source terms allow local storage.
- Keep provider-supplied facts separate from AI-derived notes.
- Treat AI mapping suggestions as review items, not automatic schema changes.
- Do not use local AI to invent missing odds, race results, runner identities, or official times.
- Keep deterministic model scoring separate from language-model explanations.
