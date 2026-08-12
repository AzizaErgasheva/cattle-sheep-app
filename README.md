# 🐄 Cow vs. Sheep Classifier — Full-Stack ML Portfolio App

An end-to-end computer vision app: upload a photo, pick a trained model, get a
prediction with confidence, see a **Grad-CAM** heatmap explaining *why* the
model decided that, browse past predictions, and compare all four models'
test-set performance on a live dashboard.

Built on top of the models trained in the companion notebook (custom CNN →
ResNet50 → MobileNetV2 → EfficientNetB0), this repo wraps them in a
production-shaped service: a FastAPI backend in clean-architecture style and
a React/TypeScript frontend, containerized and ready to deploy.

---

## 🎥 Demo Video

> _Video coming soon — add your walkthrough here._
>
> To embed it: drag and drop the video file directly into this README on
> GitHub.com (in the web editor) — GitHub will upload it and auto-insert the
> correct link. Or paste a YouTube link below and it'll render as a clickable
> thumbnail:
>
> [![Demo video](https://img.shields.io/badge/▶-Watch_the_demo-22d3ee?style=for-the-badge)](#)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Comparison](#model-comparison)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
  - [Quickest path: Docker](#quickest-path-docker)
  - [Manual local dev](#manual-local-dev)
- [Testing](#testing)
- [Deployment](#deployment)
- [What's Verified](#whats-verified-vs-what-you-still-need-to-do)
- [Roadmap](#roadmap--what-id-improve-next)

---

## Features

| Feature | Where |
|---|---|
| Model selector (ResNet50 / MobileNetV2 / EfficientNetB0 / Custom CNN) | Classify tab |
| Prediction + confidence + per-class probabilities | Classify tab |
| Grad-CAM explanation, generated on demand ("why this prediction?") | Classify tab |
| Prediction history (SQLite-backed, persists across restarts) | History tab |
| Live analytics dashboard (accuracy / precision / recall / F1 per model) | Dashboard tab |

---

## Architecture

```
[ React frontend ]  --HTTP-->  [ FastAPI backend ]  --loads-->  [ .keras models ]
```

Two deployable units: a static frontend and a Python inference API, in one
monorepo. The backend follows **clean architecture** — business logic (what a
prediction *is*, how to run inference) is isolated from frameworks (FastAPI,
Keras, React), so any layer can be swapped without touching the others.

| Layer | Responsibility | Depends on |
|---|---|---|
| **Domain** | `Prediction`, `ImageInput`, `ModelSummary`, `HistoryEntry` entities; `ClassifierPort` / `ExplainerPort` / `ModelRegistryPort` interfaces | Nothing — pure Python |
| **Application** | Use cases: `PredictImageUseCase`, `ExplainPredictionUseCase`, history use cases, `ListModelsUseCase` | Domain interfaces only |
| **Infrastructure** | `KerasModelAdapter` (loads a `.keras` model, predicts + Grad-CAM), `ModelRegistry` (auto-discovers models, merges metadata), `SqliteHistoryRepository`, image utils | Domain interfaces (implements them) + TensorFlow/Keras/Pillow |
| **API** | FastAPI routes, Pydantic schemas, DI wiring, error → HTTP translation | Application layer only |

Splitting Application from Infrastructure means every use case is
unit-tested against a **fake** registry/classifier — no TensorFlow, no GPU,
fast CI.

### Why one `KerasModelAdapter` per model, not two separate classes

`ModelHandle` combines the classifier and explainer interfaces on one object,
so a model's weights are loaded once and shared between `/predict` and
`/predict/explain` for that model, instead of two separate copies of the same
network living in memory.

### Grad-CAM, briefly

The last `Conv2D` layer is located automatically. For flat models (custom
CNN) a single graph runs a forward + backward pass. For models wrapping a
nested backbone (ResNet50 / MobileNetV2 / EfficientNetB0), a self-contained
graph is built from the backbone's own input/output and the classification
head is replayed manually inside the same gradient tape — necessary because
Keras won't let a single `Model` span into a nested submodel's internals
directly.

---

## Tech Stack

**Backend:** Python 3.11, FastAPI, Pydantic v2, TensorFlow (CPU), Pillow,
SQLite (stdlib `sqlite3`), pytest, Docker.

**Frontend:** React 18, Vite, TypeScript (strict), Tailwind CSS, Recharts.

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── domain/          entities.py, ports.py, exceptions.py — zero framework imports
│   │   ├── application/      predict_use_case.py, explain_use_case.py, history_use_cases.py, list_models_use_case.py
│   │   ├── infrastructure/   keras_model_adapter.py, model_registry.py, history_repository.py, image_utils.py
│   │   ├── api/
│   │   │   ├── routes/        predict.py, explain.py, health.py, models.py, history.py
│   │   │   ├── schemas.py     Pydantic request/response models
│   │   │   ├── dependencies.py DI providers
│   │   │   └── main.py         FastAPI app + CORS + router wiring
│   │   └── config.py          Settings (APP_* env vars)
│   ├── models/                 metadata.json (+ your .keras files, gitignored)
│   ├── scripts/download_models.py   pulls weights from Hugging Face Hub at container start
│   ├── tests/unit/ + tests/integration/
│   ├── entrypoint.sh, Dockerfile, requirements.txt
│   └── data/                    SQLite history.db lives here at runtime
├── frontend/
│   ├── src/
│   │   ├── api/client.ts        typed fetch wrapper, mirrors backend schemas
│   │   ├── components/          UploadDropzone, ModelSelector, PredictionCard, GradCamOverlay, HistoryPanel, Dashboard
│   │   ├── App.tsx               tab navigation: Classify / History / Dashboard
│   │   └── main.tsx, index.css
│   ├── Dockerfile, package.json, vite.config.ts, tailwind.config.js
├── docker-compose.yml
└── README.md
```

---

## Model Comparison

Test-set results from the training notebook (`backend/models/metadata.json`):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **ResNet50** ★ best | 97.69% | 97.69% | 97.69% | 97.69% |
| EfficientNetB0 | 96.72% | 96.73% | 96.72% | 96.72% |
| MobileNetV2 | 96.34% | 96.42% | 96.34% | 96.34% |
| Custom CNN | 82.27% | 83.10% | 82.27% | 82.19% |

ResNet50 is the most accurate but also the heaviest to serve. If cold-start
latency matters on a free hosting tier, MobileNetV2 is a reasonable
trade — 96.3% vs. 97.7% accuracy for a much smaller, faster container.

---

## API Reference

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/health` | GET | — | `{"status": "ok", "model": "resnet50"}` |
| `/models` | GET | — | All available models + metrics, for the selector/dashboard |
| `/predict` | POST | multipart `file` + optional `model_name` | `{label, confidence, probabilities, model_name}` — also recorded to history |
| `/predict/explain` | POST | multipart `file` + optional `model_name` | Grad-CAM overlay PNG, prediction fields in `X-Predicted-Label` / `X-Confidence` / `X-Model-Name` headers |
| `/history` | GET | `?limit=` (default 20, max 200) | Recent predictions, newest first |
| `/history` | DELETE | — | Clears history, `204 No Content` |

Interactive docs at `http://localhost:8000/docs` once the backend is running.

---

## Getting Started

### Quickest path: Docker

1. Copy your trained models into the backend, one `.keras` file per
   architecture — you don't need all four, the registry only offers
   whichever ones exist:
   ```
   backend/models/resnet50.keras
   backend/models/mobilenet.keras
   backend/models/efficientnet.keras
   backend/models/custom_cnn.keras
   ```
   `backend/models/metadata.json` is already included, pre-filled with the
   notebook's actual results.

2. Run both services:
   ```bash
   docker compose up --build
   ```

3. Open `http://localhost:4173`. Backend API docs at `http://localhost:8000/docs`.

### Manual local dev

```bash
# terminal 1 — backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload

# terminal 2 — frontend
cd frontend
npm install
cp .env.example .env   # point VITE_API_URL at your backend
npm run dev
```

Frontend dev server: `http://localhost:5173`.

---

## Testing

```bash
cd backend && pytest -v
```

Domain/application layers are tested directly against fakes; API routes via
FastAPI's `dependency_overrides`; the model registry's discovery and
metadata-merging logic against real temp files with no actual model loading.
TensorFlow is only imported when a `ModelHandle` actually loads a real
`.keras` file, i.e. on a genuine `/predict` call — so the whole suite runs
fast without TensorFlow installed.

```bash
cd frontend && npm run build   # type-checks (tsc -b) then builds to dist/
```

---

## Deployment

Three parts: push code to GitHub, host model weights on Hugging Face Hub
(they're too large for git), deploy the backend, deploy the frontend.

### 1. Push to GitHub

```bash
git add .
git status   # confirm no .keras files, node_modules, or .venv are listed
git commit -m "Cow vs sheep classifier full-stack app"
git push
```

### 2. Host model weights on Hugging Face Hub

`.keras` files are gitignored on purpose — `resnet50.keras` alone is ~200MB,
over GitHub's 100MB hard file limit.

1. Rename local files if needed so they match what the backend expects:
   `resnet50.keras`, `mobilenet.keras`, `efficientnet.keras`,
   `custom_cnn.keras`, `metadata.json`.
2. Create a free **Model** repo at [huggingface.co/new](https://huggingface.co/new),
   e.g. `your-username/cow-sheep-classifier-models`.
3. Upload the 5 files via the repo's "Files" → "Add file" → "Upload files".

### 3. Deploy the backend (Render, Railway, or Fly.io)

1. New → Web Service → connect your repo.
2. **Root directory:** `backend` · **Environment:** Docker.
3. Environment variables:
   ```
   APP_HF_REPO_ID=your-username/cow-sheep-classifier-models
   APP_CORS_ALLOW_ORIGINS=https://your-frontend-domain.vercel.app
   ```
   `APP_CORS_ALLOW_ORIGINS` is a **plain comma-separated string** (e.g.
   `https://foo.com,https://bar.com`), *not* a JSON array — brackets/quotes
   will break it.
4. Deploy. First boot re-downloads ~200MB of models inside `entrypoint.sh` —
   check logs for `[download_models]` lines. If the HF repo is private, also
   set `HF_TOKEN`.
5. Confirm it's live at `https://your-backend-url/docs`.

### 4. Deploy the frontend (Vercel)

1. New Project → import the same repo. **Root directory:** `frontend`.
2. **Build command:** `npm run build` (auto-detected).
3. Environment variable (build-time — Vite bakes it in at build, not runtime):
   ```
   VITE_API_URL=https://your-backend-url.onrender.com
   ```
4. Deploy → you'll get a URL like `https://cow-sheep-app.vercel.app`.

### 5. Close the loop: fix CORS

Update the backend's `APP_CORS_ALLOW_ORIGINS` to the real frontend URL and
redeploy. Without this, the frontend loads but every `/predict` call fails
with a CORS error in the browser console.

### Sanity check

Open the live frontend, upload a cow or sheep photo, confirm a prediction
comes back, and check the History and Dashboard tabs load too.

### Free-tier notes

- Render's free web services sleep after inactivity and take ~30–60s to wake
  — the first prediction after idle time will be slow, not broken.
- Render's free tier has an ephemeral filesystem, so every restart re-runs
  `entrypoint.sh` and re-downloads all ~260MB of models. If that becomes a
  problem, look into a persistent disk add-on.

---

## What's Verified vs. What You Still Need to Do

**Verified:**
- Backend: test suite passes, OpenAPI schema builds cleanly across all
  endpoint groups, every file byte-compiles, the model registry's discovery
  and metadata-merging logic is tested with real temp files, adapters fail
  gracefully (no TensorFlow import) when a model file is missing.
- Frontend: `npm run build` (type-check + Vite build) succeeds with zero
  TypeScript errors, dev server boots cleanly.

**Not yet verified (needs real model files):**
- A real end-to-end request through `/predict` and `/predict/explain`
  against actual `.keras` weights. Once your models are in place, hit
  `/docs` (or the Classify tab) to try a real image.
- Preprocessing is inferred by filename: `resnet50`, `mobilenet`, and
  `efficientnet` get the matching `preprocess_input`; anything else (like
  `custom_cnn`) is treated as a flat model with no preprocessing. Rename
  files to match, or the wrong preprocessing will get applied silently.

---

## Roadmap / What I'd Improve Next

- Multi-class support beyond cow/sheep.
- Active learning loop on misclassified images from the history log.
- k-fold cross-validation results alongside the single test-set split.
- Frontend component tests (Vitest + React Testing Library) for
  `PredictionCard` and friends.
