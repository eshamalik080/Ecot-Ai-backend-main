# ECOT AI Backend

FastAPI backend for real-time electricity monitoring, anomaly/theft detection, and cost-aware energy forecasting.

The service ingests live PZEM readings from Firebase Realtime Database, stores and processes them in PostgreSQL, applies rule-based plus ML checks, and exposes device insights through REST APIs.

## Features

- Real-time ingestion from Firebase Realtime Database on a scheduler.
- Rule-based anomaly checks (voltage/current/power thresholds and spike detection).
- Isolation Forest model training plus inference for anomaly scoring.
- Theft risk scoring and suspicious event logging.
- Hourly summaries with automatic retention cleanup.
- 24-hour energy forecast with tariff-aware optimization tips.
- REST API with Swagger docs at `/docs`.
- Docker Compose setup for API and PostgreSQL.

## Tech Stack

- Python 3.11
- FastAPI plus Uvicorn
- SQLAlchemy plus PostgreSQL
- APScheduler
- Firebase Admin SDK
- scikit-learn, pandas, numpy
- pytest

## Project Structure

```text
app/
  api/          # Route handlers
  models/       # SQLAlchemy ORM models
  schemas/      # Pydantic response schemas
  services/     # Ingestion, baseline, cleanup, forecasting logic
  tasks/        # Scheduler and job orchestration
  ml/           # Feature engineering, anomaly, theft, tariff modules
  main.py       # FastAPI app entrypoint
scripts/
  train.py      # Manual model training entrypoint
docker/
  Dockerfile    # API image
  postgres/
    init.sql    # Postgres initialization
tests/          # Unit tests
```

## Prerequisites

- Python 3.11+
- PostgreSQL 16+ (or Docker)
- Firebase service account JSON key
- Access to Firebase Realtime Database URL

## Environment Configuration

1. Copy environment template:

```bash
cp .env.example .env
```

2. Update required values in `.env`:

- `DATABASE_URL`
- `FIREBASE_CREDENTIALS_PATH`
- `FIREBASE_DATABASE_URL`

3. Place your Firebase credentials file at the configured path (default: `firebase_credentials.json`).

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open docs:

- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Run with Docker Compose

```bash
docker compose up --build
```

Services:

- API: [http://localhost:8000](http://localhost:8000)
- PostgreSQL: `localhost:5432`

Default DB credentials from `docker-compose.yml`:

- user: `ecot`
- password: `ecot_pass`
- database: `ecot_db`

## API Overview

Base prefix: `/api/v1`

System and model:

- `GET /health`
- `GET /model/status`
- `POST /model/retrain`
- `GET /scheduler/jobs`
- `POST /scheduler/trigger/{job_id}`

Device data:

- `GET /devices`
- `GET /devices/{device_id}/result`
- `GET /devices/{device_id}/readings?limit=50`
- `GET /devices/{device_id}/summaries?limit=24`
- `GET /devices/{device_id}/forecast`

Events and forecasting:

- `GET /events/suspicious`
- `POST /forecast/trigger`

## Model Training

Manual training from stored `raw_readings`:

```bash
python -m scripts.train
```

You can also trigger retraining through API:

```bash
curl -X POST http://localhost:8000/api/v1/model/retrain
```

## Scheduler Jobs

Jobs are started at application startup:

- `ingestion_job` (Firebase polling)
- `cleanup_job` (data retention cleanup)
- `baseline_job` (baseline rebuild)
- `forecasting_job` (24h forecast regeneration)

Intervals are controlled via `.env` settings.

## Testing

Run unit tests:

```bash
pytest -q
```

## Common Troubleshooting

- **DB connection errors**: verify `DATABASE_URL` and Postgres availability.
- **Firebase init failures**: confirm credentials path and database URL.
- **No model loaded warning**: expected on fresh setup; run training once data exists.
- **No forecast found**: wait for `forecasting_job` or trigger `POST /forecast/trigger`.

## Security Notes

- Never commit `.env` or Firebase credential files.
- Restrict CORS origins for production.
- Use managed secrets in deployment environments.

## License

Add your preferred license in this repository (for example, MIT or Apache-2.0).
