# SETUP: Standbai AI Service

## Prerequisites
- **Python 3.12** (recommended). The fire-scorer pickle was made with scikit-learn 1.6.1, which has no wheel for Python 3.14. On 3.14, a newer scikit-learn loads it with an `InconsistentVersionWarning`.
- pip and `venv`.
- **Keys:**
  - for the LLM features, a **Featherless API key** (ForgeHacks perk; see below);
  - optionally a **Gemini API key** (Google AI Studio) for the fallback.

## Install and run

```bash
git clone https://github.com/ITHACKITS/AI-FORGEHACK-2026.git
cd AI-FORGEHACK-2026
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload --env-file .env
```

- Service: http://localhost:8001
- Interactive docs: http://localhost:8001/docs

## Environment variables

**Read by the code on `main` today:**

| Variable | Default | Purpose |
|---|---|---|
| `UDARASIAGA_OFFLINE` | `1` | `1` blocks network calls (engine copy and Gemini) |
| `GEMINI_API_KEY` | - | Used by the current advisory code (to be replaced by the gateway) |
| `FIRMS_MAP_KEY`, `OPENAQ_API_KEY` | - | Engine copy (not needed for the web app) |

**Added by the PRD tickets** (see `PRD.md` §4.2): `LLM_PRIMARY`, `FEATHERLESS_API_KEY`, `FEATHERLESS_BASE_URL`, `LLM_MODEL_FAST`, `LLM_MODEL_LARGE`, `GEMINI_MODEL`, `LLM_TIMEOUT_S`, `AI_INTERNAL_TOKEN`, `MODEL_DIR`.

Never commit `.env`. **Rotate any key that was ever pasted into a chat.**

## Get a Featherless key (ticket AI-00)
1. Open the ForgeHacks participant packet or Discord for the Featherless sign-up link (the promo code is applied automatically).
2. Create an API key in the Featherless dashboard.
3. Put it in `.env` as `FEATHERLESS_API_KEY`.
4. Check that it works: `curl https://api.featherless.ai/v1/models -H "Authorization: Bearer $FEATHERLESS_API_KEY" | head`.

## Tests and lint

```bash
UDARASIAGA_OFFLINE=1 python -m pytest -q   # 9 passed on 2026-10-10
ruff check .
python test_fire_ai.py                      # manual check of the fire scorer (moves into tests/ in AI-08)
```

## Training

- **Existing detection scorer:** `kaggle_train.py` runs on Kaggle with the NASA FIRMS dataset attached (`/kaggle/input`), writes `fire_predictor_global.pkl`, and you copy it to `app/models/`. `train_fire_model.py` is stale (AI-08).
- **Fire-activity outlook (AI-06):** `app/outlook/train.py` (to be written) aggregates FIRMS to cell × week, trains with a time-based split, writes `app/models/outlook_v1.joblib`, the cell and climatology Parquet files, and `reports/outlook_metrics.json`. Run it on Kaggle where the data lives, and commit only the small artifacts.
- **Never commit raw data:** `data/*.zip` and `data/*.csv` are git-ignored.

## Deploy to Google Cloud Run (ticket AI-09, Hikari)

```bash
gcloud run deploy standbai-ai --source . --region asia-southeast2 \
  --memory 1Gi --min-instances 1 --max-instances 2 --allow-unauthenticated \
  --set-env-vars LLM_PRIMARY=featherless,GEMINI_MODEL=gemini-flash-latest,LLM_TIMEOUT_S=8 \
  --set-secrets FEATHERLESS_API_KEY=featherless-api-key:latest,GEMINI_API_KEY=gemini-api-key:latest,AI_INTERNAL_TOKEN=ai-internal-token:latest
```

- Add a `Dockerfile` with `FROM python:3.12-slim`, following the pattern in the backend PRD §4.10.
- Every `/api/v1/*` route must check `X-Internal-Token` (AI-01), so only the backend can use it.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `InconsistentVersionWarning` when loading the pickle | Use Python 3.12 with `scikit-learn==1.6.1`, or re-export the model with your scikit-learn |
| `No module named 'numpy._core'` | Install numpy 2.x |
| `/api/v1/advisory` returns 500 (unexpected keyword argument) | Known bug: wrong function arguments (AI-02) |
| Featherless returns 429 | Concurrency limit reached; the gateway falls back to Gemini (AI-01) |
| Gemini calls fail with 404 | The old code uses shut-down model ids; set `GEMINI_MODEL=gemini-flash-latest` after AI-01 |
