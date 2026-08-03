# Project Context

This file captures the important context imported from the original `Open horse_predictor repo` task so future work stays attached to the same project direction.

## Imported Decisions

- The repo is `liamryan391/horse_predictor`.
- The local checkout lives at `C:\Users\liamr\Documents\Codex\2026-05-01\could-you-open-my-github-repo\horse_predictor`.
- The original pandas error happened because VS Code launched `c:\python313\python.exe` instead of the repo virtual environment.
- MySQL was chosen over T-SQL for the production database path.
- SQLite remains only as a local/demo fallback.
- SQLAlchemy is the storage abstraction.
- FastAPI is the backend API direction.
- React and TypeScript are the frontend direction.
- The React app should have top navigation with Home, About Us, and the main tool page named Race Lab.
- The Streamlit app is now a legacy/internal diagnostic dashboard.
- Development notes should explain how to run the backend and frontend, and list sensible next improvements.

## Current Project State

- PR #2 is open on GitHub for the FastAPI, SQLAlchemy, React, and roadmap work.
- The active branch is `agent/add-api-react-platform`.
- `main` remains the stable branch.
- The old `codex/create-horse-racing-software-and-app` branch has already been merged into `main` through PR #1.

## Guardrails

- Do not train on `finishing_position` or other result-only fields.
- Do not present predictions as guaranteed betting advice.
- Do not expose secrets, API credentials, or production database URLs through public endpoints.
- Prefer idempotent ingestion and upserts over replacing production tables.
- Keep API provider payloads separate from internal database models.
- Keep provider-specific API mapping inside `provider_adapters.py`.
