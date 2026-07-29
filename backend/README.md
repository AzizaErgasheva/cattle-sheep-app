# Backend — Cow vs Sheep Classifier API

Clean-architecture FastAPI service with a multi-model registry, Grad-CAM
explanations, and a SQLite-backed prediction history.

## Layers

```
app/domain/          entities + interfaces, zero framework imports
app/application/      use cases, depend only on domain interfaces
app/infrastructure/   Keras model adapter, model registry, SQLite history repo
app/api/               FastAPI routes, schemas, DI wiring
```

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy your trained models into `models/`, one `.keras` file per architecture,
named to match the notebook's model keys:

```
models/resnet50.keras
models/mobilenet.keras
models/efficientnet.keras
models/custom_cnn.keras
models/metadata.json     # already included, pre-filled with the notebook's Section 10 results
```

The registry auto-discovers whatever `.keras` files are present -- you don't
need all four; it'll just offer whichever ones exist in the model selector.
`metadata.json`'s `best_model_name` decides which one is used by default when
the frontend doesn't specify a `model_name`.

## Run

```bash
uvicorn app.api.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + default model name |
| `/models` | GET | All available models + metrics, for the selector/dashboard |
| `/predict` | POST | `file` + optional `model_name` form field → prediction. Also records to history. |
| `/predict/explain` | POST | Same, plus a Grad-CAM PNG overlay |
| `/history` | GET | Recent predictions (`?limit=`), newest first |
| `/history` | DELETE | Clears history |

## Test

```bash
pytest -v
```

25 tests pass without TensorFlow installed -- domain/application layers are
tested directly, API routes via `dependency_overrides` with fakes, and the
model registry's discovery/metadata logic with real temp files but no actual
model loading. TensorFlow is only imported when a `ModelHandle`'s `_load()`
actually runs, i.e. on a real `/predict` call against a real model file.

## Docker

```bash
docker build -t cow-sheep-api .
docker run -p 8000:8000 -v $(pwd)/models:/srv/models -v $(pwd)/data:/srv/data cow-sheep-api
```

## Deploying

Your `.keras` model files are gitignored on purpose -- `resnet50.keras` alone
is ~200MB, over GitHub's 100MB hard file limit. Instead, host them on
[Hugging Face Hub](https://huggingface.co/new) (free) and let the container
download them at startup.

1. **Rename your local files first**, if you haven't -- the registry infers
   Grad-CAM preprocessing from the filename stem matching exactly `resnet50`,
   `mobilenet`, or `efficientnet`:
   ```
   resnet50.keras
   mobilenet.keras
   efficientnet.keras
   custom_cnn.keras
   metadata.json
   ```

2. **Create a Model repo on Hugging Face Hub** (huggingface.co → New Model),
   e.g. `your-username/cow-sheep-classifier-models`. Upload the 5 files above
   via the web UI's "Add file" button, or the `huggingface-cli upload`
   command.

3. **On your hosting platform** (Render, Railway, Fly.io, etc.), set these
   environment variables on the backend service:
   ```
   APP_HF_REPO_ID=your-username/cow-sheep-classifier-models
   APP_CORS_ALLOW_ORIGINS=["https://your-frontend-domain.com"]
   ```
   If the HF repo is private, also set `HF_TOKEN` to a Hugging Face access
   token with read access.

4. Deploy the `backend/` folder as a Docker service. `entrypoint.sh` runs
   `scripts/download_models.py` before starting `uvicorn` -- it downloads
   any file from the HF repo that isn't already present locally, then boots
   the server. If a model already exists locally (e.g. baked into the image
   or on a persistent volume), it's skipped, so this is safe to run on every
   deploy/restart.

Local development is unaffected either way: if `APP_HF_REPO_ID` isn't set,
`entrypoint.sh` skips the download step entirely and just starts the server
against whatever's already in `models/`.
