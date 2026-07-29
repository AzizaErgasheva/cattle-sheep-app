# Frontend — Cow vs Sheep Classifier UI

React + Vite + TypeScript + Tailwind, dark/techy theme. Talks to the backend
only through `src/api/client.ts`, whose types mirror `backend/app/api/schemas.py`.

## Setup

```bash
cd frontend
npm install
cp .env.example .env   # point VITE_API_URL at your backend
```

## Run

```bash
npm run dev
```

Visit `http://localhost:5173`. The backend must be running (see `backend/README.md`).

## Build

```bash
npm run build      # type-checks (tsc -b) then builds to dist/
npm run preview    # serve the production build locally
```

## Structure

```
src/api/client.ts             typed fetch wrapper: predict, explain, models, history
src/components/
  UploadDropzone.tsx           drag-and-drop image upload
  ModelSelector.tsx             pick which trained model to run inference with
  PredictionCard.tsx            label, confidence, per-class probability bars
  GradCamOverlay.tsx            on-demand Grad-CAM overlay
  HistoryPanel.tsx              past predictions (thumbnail, label, model, time)
  Dashboard.tsx                  recharts bar chart + table of all models' test metrics
src/App.tsx                     tab navigation: Classify / History / Dashboard
```

## Features

- **Model selector** — switch between ResNet50 / MobileNetV2 / EfficientNetB0 /
  Custom CNN per prediction; whichever models exist in the backend's `models/`
  folder show up automatically, with the best one starred.
- **Prediction history** — every `/predict` call is logged server-side
  (SQLite); the History tab lists them with thumbnails, newest first.
- **Analytics dashboard** — pulls `/models` and renders the same
  accuracy/precision/recall/F1 comparison from the notebook's Section 10,
  live in the UI.
