# Cow vs. Sheep Classifier — Full-Stack Portfolio App

End-to-end app around the models trained in the companion notebook: pick a
model, upload a photo, get a prediction with confidence, see a Grad-CAM
overlay explaining the decision, browse prediction history, and view a live
analytics dashboard of all four models' test performance.

```
.
├── backend/     FastAPI service, clean architecture (domain / application / infrastructure / api)
├── frontend/     React + Vite + TypeScript + Tailwind, dark/techy theme
└── docker-compose.yml
```

See `backend/README.md` and `frontend/README.md` for details on each half.
Architecture rationale is in `architecture.md`. **Deploying to GitHub +
production hosting: see `DEPLOY.md`.**

## Features

| Feature | Where |
|---|---|
| Model selector (ResNet50 / MobileNet / EfficientNet / Custom CNN) | Classify tab |
| Prediction + confidence + per-class probabilities | Classify tab |
| Grad-CAM explanation, generated on demand | Classify tab |
| Prediction history (SQLite-backed, persists across restarts) | History tab |
| Live analytics dashboard (accuracy/precision/recall/F1 per model) | Dashboard tab |

## Quickest path to running it

1. Copy your trained models into the backend, one `.keras` file per
   architecture (`resnet50.keras`, `mobilenet.keras`, `efficientnet.keras`,
   `custom_cnn.keras`) — you don't need all four, the registry only offers
   whichever ones exist:
   ```
   backend/models/resnet50.keras
   backend/models/mobilenet.keras
   backend/models/efficientnet.keras
   backend/models/custom_cnn.keras
   ```
   `backend/models/metadata.json` is already included, pre-filled with your
   notebook's actual Section 10 results.

2. Run both services:
   ```bash
   docker compose up --build
   ```

3. Open `http://localhost:4173`. The backend API docs are at `http://localhost:8000/docs`.

## Running without Docker (local dev)

```bash
# terminal 1
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload

# terminal 2
cd frontend
npm install
npm run dev
```

## What's verified vs. what you still need to do

Verified while building this:
- Backend: **25/25 tests pass**, OpenAPI schema builds cleanly across all 5
  endpoint groups (`/health`, `/predict`, `/predict/explain`, `/models`,
  `/history`), every file byte-compiles, the model registry's discovery and
  metadata-merging logic is tested with real temp files, adapters fail
  gracefully (no TensorFlow import) when a model file is missing.
- Frontend: `npm run build` (type-check + Vite build) succeeds with zero
  TypeScript errors, dev server boots cleanly.

Not yet verified (needs your model files, which weren't available while building):
- A real end-to-end request through `/predict` and `/predict/explain` against
  actual `.keras` weights. Once you drop your models in, run the backend and
  hit `/docs` (or the frontend's Classify tab) to try a real image.
- Preprocessing is inferred by filename: `resnet50`, `mobilenet`, and
  `efficientnet` get the matching `preprocess_input`; anything else (like
  `custom_cnn`) is treated as a flat model with no preprocessing. If you use
  different filenames, rename your `.keras` files to match, or the wrong
  preprocessing will get applied silently.
