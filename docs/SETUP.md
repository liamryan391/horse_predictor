# Developer Setup

Horse Predictor supports three local development paths:

- Windows PowerShell with the repo virtual environment.
- Linux/macOS shell with the repo virtual environment.
- Docker Compose with MySQL, FastAPI, and React services.

## Target Stack

- React and TypeScript frontend.
- FastAPI Python backend.
- SQLAlchemy data-access layer.
- MySQL production database.
- SQLite local/demo fallback.
- Streamlit diagnostic dashboard.

## Configuration

Copy the example environment file before custom local work:

```powershell
Copy-Item .env.example .env
```

On Linux/macOS:

```bash
cp .env.example .env
```

Important settings:

- `APP_ENV`: `development`, `test`, `staging`, or `production`.
- `DATABASE_URL`: SQLAlchemy database URL. SQLite is fine locally; staging and production should use MySQL.
- `BACKEND_CORS_ORIGINS`: comma-separated frontend origins allowed by FastAPI.
- `VITE_API_BASE_URL`: frontend API base URL for deployed builds.
- `HORSE_API_PROVIDER`: `sample`, `generic`, `theracingapi`, or `ourhub`.

The backend validates staging/production config at startup. In deployed environments, SQLite is rejected so production work does not accidentally run against a local file database.

## Windows PowerShell

From the repo root:

```powershell
.\scripts\start-dev.ps1
```

That script creates `.venv` if needed, installs Python and frontend dependencies, seeds sample data, then starts:

- FastAPI: `http://127.0.0.1:8000`
- React: `http://127.0.0.1:5173`

For a faster restart after dependencies are installed:

```powershell
.\scripts\start-dev.ps1 -SkipInstall
```

## Linux And macOS

From the repo root:

```bash
chmod +x scripts/start-dev.sh
./scripts/start-dev.sh
```

That script creates `.venv` if needed, installs dependencies, seeds sample data, then starts FastAPI and React.

## Manual Local Commands

Backend setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe data_pipeline.py --provider sample
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Frontend setup:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Linux/macOS equivalents:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m alembic upgrade head
.venv/bin/python data_pipeline.py --provider sample
.venv/bin/python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

## Docker Compose

Docker Compose starts MySQL, FastAPI, and React together:

```powershell
docker compose up --build
```

Open:

```text
http://127.0.0.1:5173
```

The Compose stack uses MySQL with these local-only defaults:

- Database: `horse_predictor`
- User: `horse_user`
- Password: `horse_password`

Do not reuse those values for staging or production.

## VS Code

Open VS Code at the repo root:

```powershell
code .
```

The repo includes `.vscode/settings.json`, which points Python tooling at `.venv\Scripts\python.exe`.
