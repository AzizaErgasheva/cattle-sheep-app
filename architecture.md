# Cow vs. Sheep Classifier — Web App Architecture

A portfolio web app that lets a visitor upload a photo, get a prediction (cow/sheep,
confidence, per-class probability) and a Grad-CAM overlay showing *why* the model
decided that — built on top of the trained `best_model.keras` from the notebook.

Goal: show off the whole pipeline (data → model → serving → UI), not just a demo.
The architecture below is deliberately "clean architecture" style: business logic
(what a prediction *is*, how to run inference) is isolated from frameworks (FastAPI,
Keras, React), so any layer can be swapped without touching the others — a good
thing to point at in an interview.

---

## 1. High-level shape

```
[ React frontend ]  --HTTP-->  [ FastAPI backend ]  --loads-->  [ best_model.keras ]
```

Two deployable units: a static frontend and a Python inference API. Monorepo,
two top-level folders.

## 2. Backend — layers (inside → out dependency direction)

| Layer | Responsibility | Knows about |
|---|---|---|
| **Domain** | `Prediction`, `ImageInput` entities; `ClassifierPort` / `ExplainerPort` interfaces | Nothing else — pure Python |
| **Application** | Use cases: `PredictImageUseCase`, `ExplainPredictionUseCase` | Domain only (depends on interfaces, not implementations) |
| **Infrastructure** | `KerasClassifierAdapter`, `GradCamAdapter`, file/image utils | Domain interfaces (implements them) + TensorFlow/Keras |
| **API (interface)** | FastAPI routes, Pydantic request/response schemas, error handling | Application layer only |

This is the same shape shown in the diagram above. The point of splitting
Application from Infrastructure: your use case (`PredictImageUseCase`) can be
unit-tested with a fake classifier, with zero TensorFlow involved — fast, no GPU
needed for CI.

### Folder structure

```
backend/
  app/
    domain/
      entities.py          # Prediction, ImageInput (dataclasses)
      ports.py              # ClassifierPort, ExplainerPort (ABCs / Protocols)
    application/
      predict_use_case.py
      explain_use_case.py
    infrastructure/
      keras_classifier.py   # implements ClassifierPort, loads best_model.keras
      gradcam_explainer.py  # implements ExplainerPort
      image_utils.py        # decode/resize/normalize
    api/
      routes/
        predict.py           # POST /predict
        explain.py           # POST /predict/explain
        health.py            # GET /health
      schemas.py             # Pydantic models
      main.py                # FastAPI app, DI wiring
    config.py                # settings (model path, img size, threshold)
  models/
    best_model.keras
    metadata.json             # from your notebook's export step
  tests/
    unit/                     # test use cases with fake ClassifierPort
    integration/               # test /predict with a real image, real model
  Dockerfile
  requirements.txt
```

### API design

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/health` | GET | — | `{"status": "ok", "model": "resnet50"}` |
| `/predict` | POST | multipart image | `{"label": "cow", "confidence": 0.98, "probs": {"cow": 0.98, "sheep": 0.02}}` |
| `/predict/explain` | POST | multipart image | Grad-CAM overlay image (PNG) + same prediction fields |
| `/model/info` | GET | — | model name, test accuracy/F1 from `metadata.json`, image size |

Keep `/predict` and `/predict/explain` separate — Grad-CAM is slower (needs a
backward pass), so the plain prediction should stay fast for a snappy UI, with
the explanation fetched on demand (e.g. "why?" button).

### Dependency injection

Wire the concrete adapter into the use case once, in `main.py`:

```python
classifier = KerasClassifierAdapter(model_path=settings.MODEL_PATH)
predict_use_case = PredictImageUseCase(classifier=classifier)
```

Routes call `predict_use_case.execute(image)` — they never import Keras directly.
This is what makes the unit tests fast and the swap-the-model story real.

## 3. Frontend

```
frontend/
  src/
    api/
      client.ts              # fetch wrapper, calls backend
    components/
      UploadDropzone.tsx
      PredictionCard.tsx      # label + confidence + prob bar
      GradCamOverlay.tsx      # side-by-side original vs heatmap
      ModelInfoBadge.tsx      # "ResNet50 · 97.7% test accuracy"
    pages/
      Home.tsx
    App.tsx
  index.html
```

React + Vite + Tailwind is a reasonable default — it's fast to set up and reads
as competent without over-engineering for a portfolio piece. Plain HTML/JS is
also fine if you want zero build tooling; the API contract doesn't care.

Suggested UI flow: drag-and-drop or file picker → immediate `/predict` call →
show label + confidence bar → optional "Why?" button → calls `/predict/explain`
→ shows the Grad-CAM overlay next to the original image. Also show a static
"Model comparison" section on the page (your Section 10 table) — it's good
portfolio content and costs nothing at runtime.

## 4. Testing (matches the "test properly" theme from the notebook)

- **Unit** — `PredictImageUseCase` tested against a fake `ClassifierPort` that
  returns fixed probabilities. No model loading, runs in milliseconds.
- **Integration** — one test that hits `/predict` with a real sample image and
  the real `best_model.keras`, asserts the response shape and a sane label.
- **Frontend** — component tests for `PredictionCard` rendering given a mock
  API response (Vitest/React Testing Library).

## 5. Deployment

- **Backend**: Docker image (Python slim + TensorFlow CPU), deployed to
  Render / Railway / Fly.io, or Hugging Face Spaces (Gradio/FastAPI template) —
  the easiest option for an ML portfolio piece since it's free and expects
  exactly this kind of app.
- **Frontend**: static build deployed to Vercel/Netlify, calling the backend
  API via `VITE_API_URL`.
- **Model size note**: ResNet50 is your best model by accuracy, but it's the
  heaviest to serve. If cold-start latency matters on a free tier, consider
  shipping MobileNet instead (96.3% vs. 97.7% — a small accuracy trade for a
  much smaller, faster container) and say so explicitly in the README as a
  deliberate engineering trade-off. That's a good line for an interview.

## 6. README structure for the portfolio

1. One-line pitch + live demo link + architecture diagram (the one above)
2. Problem framing: binary classification, dataset, why cow/sheep
3. Model comparison table (your Section 10 output) — this is your strongest
   evidence of rigor, lead with it
4. Architecture explanation (link to this doc or inline summary)
5. How to run locally (`docker-compose up`)
6. What you'd improve next (multi-class, active learning on misclassified
   images, k-fold results if you kept that section)
