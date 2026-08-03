# Live Provider Check - August 3, 2026

This check used local `.env` credentials without printing secrets.

## Result Summary

- OurHub credentials work.
- The redacted provider smoke command passes for OurHub.
- OurHub `course-info/2026-08-03` and `runner-info/2026-08-03` returned live race-card data.
- OurHub imported 508 current runners into a fresh SQLite check database.
- Imported tracks were normalized to 7 real tracks: Clairefontaine, Cork, Naas, Nottingham, Ripon, Vichy, and Windsor.
- The API found meetings, filtered Windsor correctly, returned zero rows for a fake track, and produced ranked Windsor predictions.
- The Racing API credentials work for `/v1/courses`, but the account plan does not currently allow racecards or results.

## OurHub Field Coverage

OurHub live runner import coverage in the check database:

- `track`: 100%
- `distance`: 100%
- `surface` / going: 100%
- `horse`: 100%
- `jockey`: 100%
- `trainer`: 100%
- `horse_weight`: 100%
- `draw`: 96.3%
- `class_rating`: 47.4%
- `owner`: 0%
- `odds`: 0%

The missing owner and odds fields are provider-depth gaps, not ingestion failures.

## The Racing API Account Status

The Racing API HTTP Basic Authentication is valid for account-level calls such as `/v1/courses`.

Current plan limits found during testing:

- `/v1/racecards`: `Basic Plan required`
- `/v1/results`: `Standard Plan required`

This means The Racing API cannot currently provide live racecards or historical results to the app until the account plan is enabled or upgraded.

The redacted smoke command confirms the same plan boundary without exposing credentials:

```text
courses: 200, 979 rows
racecards: 401, Basic Plan required
results: 401, Standard Plan required
```

## App Check

Fresh DB used for the clean app check:

```text
.codex_tmp/live-ourhub-app-20260803-1751.db
```

Clean DB contents:

- historical rows: 6 sample rows
- current rows: 511 total
- OurHub current rows: 508
- sample current rows: 3

API checks passed:

- `/api/v1/health`: `ok`
- `/api/v1/meetings`: 8 tracks, including the 7 OurHub tracks plus sample York
- `/api/v1/race-card?track=Windsor`: 49 rows and all rows matched Windsor
- `/api/v1/race-card?track=NotARealTrack`: 0 rows
- `/api/v1/predictions?track=Windsor`: 49 ranked rows
- `/api/v1/data-quality`: provider freshness reports OurHub success

After the provider smoke command was added, the real import path was rechecked in a fresh `.codex_tmp` database. The API direct-call check again returned 8 meetings, 49 Windsor race-card rows, 49 Windsor prediction rows, zero fake-track rows, and 3 ingestion entries.

Monitoring was `critical` because the live OurHub card differs heavily from the tiny sample historical reference set. That is expected until real historical results are imported from a licensed provider.

## Code Fixes From The Check

- OurHub runner labels are now split into course, off time, and race name before mapping to course metadata.
- Built-in providers now use their official default base URLs even when a stale `HORSE_API_BASE_URL` is present.
- Provider HTTP failures now include a short response detail in ingestion logs, for example account-plan messages.
- `scripts/provider-smoke-check.py` now provides a reusable no-secret provider certification command.

## Next Roadmap Decision

OurHub is good enough for live race-card display and track matching.

It is not enough by itself for full model certification because it does not provide odds or historical results in the current adapter flow. The next roadmap work should prioritize:

- avoiding paid provider lock-in until the app proves value
- building a local data broker API for OurHub, manual files, caches, and future provider plugins
- adding optional Ollama/LM Studio/llama-cpp-python support for payload review and explanations
- adding an odds-capable provider or exchange feed
- enabling The Racing API Basic/Standard plan or choosing another historical-results provider
- adding provider stable IDs into normalized race, runner, and odds tables
- widening historical training data before treating drift/model quality as production evidence
