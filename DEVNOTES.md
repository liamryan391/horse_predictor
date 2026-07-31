# Dev Notes

## Why `ModuleNotFoundError: No module named 'pandas'` happened

VS Code ran the system Python:

```powershell
c:\python313\python.exe
```

The project dependencies are installed in the repo virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

Open VS Code at the repo folder, not at the VS Code install folder:

```powershell
cd C:\Users\liamr\Documents\Codex\2026-05-01\could-you-open-my-github-repo\horse_predictor
code .
```

The repo includes `.vscode/settings.json` so VS Code should select `.venv\Scripts\python.exe`.

## Fresh setup

```powershell
cd C:\Users\liamr\Documents\Codex\2026-05-01\could-you-open-my-github-repo\horse_predictor
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Seed the database:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider sample
```

## Run the backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
```

## Run the React frontend

PowerShell blocks `npm.ps1` on some machines, so use `npm.cmd`:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://127.0.0.1:5173
```

## MySQL setup

MySQL is the chosen production database path. SQLite remains useful for local demos, but real ingestion and model learning should use MySQL.

Create a database:

```sql
CREATE DATABASE horse_predictor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'horse_user'@'localhost' IDENTIFIED BY 'replace_this_password';
GRANT ALL PRIVILEGES ON horse_predictor.* TO 'horse_user'@'localhost';
FLUSH PRIVILEGES;
```

Set the connection string:

```powershell
$env:DATABASE_URL="mysql+pymysql://horse_user:replace_this_password@localhost:3306/horse_predictor"
```

Then run:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider sample
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

## API ingestion

Sample data:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider sample
```

OurHub race cards:

```powershell
$env:HORSE_API_KEY="your_ourhub_key"
.\.venv\Scripts\python.exe data_pipeline.py --provider ourhub --days-ahead 0
```

The Racing API:

```powershell
$env:RACING_API_USERNAME="your_username"
$env:RACING_API_PASSWORD="your_password"
.\.venv\Scripts\python.exe data_pipeline.py --provider theracingapi
```

Hourly worker:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider theracingapi --repeat-hourly
```

## Improvements to do next

- Move scheduled ingestion into a proper background worker such as APScheduler, Celery, or a hosted cron job.
- Add data validation before database writes, especially for provider-specific API payloads.
- Store trained model metrics and backtests in SQL instead of retraining only on request.
- Add authentication before exposing the tool outside localhost.
- Add model evaluation: log loss, calibration, profit simulation, and race-level holdout testing.
- Add a managed deployment target: MySQL, FastAPI service, React static hosting, and a scheduled ingestion job.
- Replace sample-derived features with richer provider fields such as going, class, official rating, recent form, trainer strike rate, and market movement.
