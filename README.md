

https://github.com/user-attachments/assets/7f05ec15-b900-4284-8cac-19ce9f41bc96





# 🐄 Cow vs. Sheep Classifier

A binary image classifier that tells cattle from sheep — trained on a cleaned
subset of the Animals-10 dataset, benchmarked across four architectures, and
shipped as a full-stack app with Grad-CAM explanations, prediction history,
and a live model-comparison dashboard.

---

## Table of Contents

1. [Problem](#1-problem)
2. [Dataset](#2-dataset)
3. [Methodology](#3-methodology)
4. [Architecture](#4-architecture)
5. [Training](#5-training)
6. [Results](#6-results)
7. [Error Analysis](#7-error-analysis)
8. [Demo](#8-demo)
9. [Installation](#9-installation)
10. [Future Improvements](#10-future-improvements)

---

## 1. Problem

Binary image classification: given a photo, decide whether it shows a **cow**
or a **sheep**. Framed intentionally narrow — two visually similar livestock
classes — to make the project a clean testbed for comparing a from-scratch
CNN against transfer learning, and for building real interpretability
(Grad-CAM) and evaluation (confusion matrices, McNemar's test) around the
result, rather than just reporting a single accuracy number.

The end goal is a working end-to-end system: notebook → trained model →
served API → UI a non-technical person could actually use.

---

## 2. Dataset

**Source:** [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
(Kaggle, Alessio Corradi) — a 10-class animal image dataset. Only two classes
were used: `mucca` (cow) and `pecora` (sheep).

| Stage | Cow | Sheep | Total |
|---|---|---|---|
| Raw (Animals-10 subset) | 1,866 | 1,820 | 3,686 |
| After cleaning | 1,760 | 1,698 | 3,458 |

### AI-assisted cleaning

Animals-10 is known to contain mislabelled images — the cow folder includes
goats and deer, the sheep folder includes goats, llamas, and other animals.
Training on these corrupts both the learned features and the evaluation
metrics, so every image was passed through a pretrained MobileNetV2
(ImageNet weights) and checked for whether any of its **top-10 predicted
classes** matched the livestock category expected for that folder:

- **Cow-valid ImageNet classes:** `ox`, `water_buffalo`, `bison`
- **Sheep-valid ImageNet classes:** `ram`, `bighorn`, `ibex`, `llama`

Images with no match were flagged, visually spot-checked, and removed —
**106 images (5.7%)** dropped from the cow folder, **122 (6.7%)** from the
sheep folder.

### Split

Stratified 70 / 15 / 15 split (`sklearn.train_test_split`, seed 42):

| Split | Cow | Sheep | Total |
|---|---|---|---|
| Train | 1,232 | 1,188 | 2,420 |
| Validation | 264 | 255 | 519 |
| Test | 264 | 255 | 519 |

---

## 3. Methodology

- **Preprocessing:** decode → resize to 224×224 → scale to `[0, 1]`. Fed
  through a `tf.data` pipeline (`AUTOTUNE` parallel loading + prefetch).
- **Augmentation** (training split only): random horizontal flip, brightness
  (±0.15), contrast (0.8–1.2×), hue (±0.05), saturation (0.8–1.2×), then
  clipped back to `[0, 1]`.
- **Four models trained and compared** on the identical data pipeline:
  a custom CNN trained from scratch (performance floor), plus three
  ImageNet-pretrained backbones fine-tuned for this task — **ResNet50**,
  **MobileNetV2**, **EfficientNetB0**.
- **Output:** single sigmoid unit, binary cross-entropy loss, threshold 0.5.
- **Evaluation:** held-out test set, classification report, confusion
  matrices, and a **McNemar's test** to check whether the best transfer model
  is *statistically* better than the CNN baseline — not just numerically
  higher.
- Reproducibility: global seed 42 (`random`, `numpy`, `tensorflow`) throughout.
- Trained on Kaggle, 2× Tesla T4 GPU, TensorFlow 2.20.

---

## 4. Architecture

### Model architecture

**Custom CNN** (baseline, trained from scratch):
```
Input(224,224,3)
→ [Conv2D(32→64→128→256, 3×3, no bias) → BatchNorm → ReLU → MaxPool] × 4
→ GlobalAveragePooling2D
→ Dense(256, relu) → Dropout(0.5)
→ Dense(1, sigmoid)
```
Uses BatchNorm for stable training and a cosine-decay learning-rate schedule
(instead of a fixed LR) to avoid the aggressive early decay that caused the
model to stall in earlier iterations of this project.

**Transfer-learning models** (ResNet50 / MobileNetV2 / EfficientNetB0):
```
Input(224,224,3)
→ backbone-specific preprocess_input (re-scales 0–1 input to what the backbone expects)
→ pretrained backbone (ImageNet weights)
→ GlobalAveragePooling2D → Dropout(0.3)
→ Dense(1, sigmoid)
```

### App / system architecture

```
[ React frontend ]  --HTTP-->  [ FastAPI backend ]  --loads-->  [ .keras models ]
```

The backend follows **clean architecture** — business logic (what a
prediction *is*, how to run inference) is isolated from frameworks (FastAPI,
Keras, React), so any layer can be swapped without touching the others.

| Layer | Responsibility | Depends on |
|---|---|---|
| **Domain** | `Prediction`, `ImageInput`, `ModelSummary`, `HistoryEntry` entities; `ClassifierPort` / `ExplainerPort` / `ModelRegistryPort` interfaces | Nothing — pure Python |
| **Application** | Use cases: `PredictImageUseCase`, `ExplainPredictionUseCase`, history use cases, `ListModelsUseCase` | Domain interfaces only |
| **Infrastructure** | `KerasModelAdapter` (loads a `.keras` model, predicts + Grad-CAM), `ModelRegistry` (auto-discovers models, merges metadata), `SqliteHistoryRepository`, image utils | Domain interfaces (implements them) + TensorFlow/Keras/Pillow |
| **API** | FastAPI routes, Pydantic schemas, DI wiring, error → HTTP translation | Application layer only |

**Grad-CAM implementation note:** the last `Conv2D` layer is located
automatically. For flat models (custom CNN) a single graph runs forward +
backward. For models wrapping a nested backbone, Keras can't trace a path
from the outer model's inputs directly into the backbone's internal tensors
— so a self-contained graph is built from the backbone's own input/output,
and the classification head is replayed manually inside the same gradient
tape.

```
backend/app/{domain, application, infrastructure, api}/
frontend/src/{api, components}/
docker-compose.yml
```

**API:**

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/health` | GET | — | `{"status": "ok", "model": "resnet50"}` |
| `/models` | GET | — | All available models + metrics |
| `/predict` | POST | multipart `file` + optional `model_name` | `{label, confidence, probabilities, model_name}` |
| `/predict/explain` | POST | multipart `file` + optional `model_name` | Grad-CAM overlay PNG |
| `/history` | GET / DELETE | `?limit=` | Recent predictions / clear |

---

## 5. Training

| Model | Optimizer / LR | Epochs (max) | Callbacks |
|---|---|---|---|
| Custom CNN | Adam, cosine decay from 1e-3 | 40 | EarlyStopping (patience 8, monitor `val_loss`) |
| ResNet50 / MobileNetV2 / EfficientNetB0 — **Phase 1** (frozen backbone) | Adam, 1e-3 | 10 | EarlyStopping (patience 4) + ReduceLROnPlateau (factor 0.5, patience 2) |
| Same — **Phase 2** (fine-tune) | Adam, 1e-5 | 15 | EarlyStopping (patience 6) |

Phase 2 unfreezes the **top 30 layers** of each backbone, letting it adapt
generic ImageNet features to the specific visual cues that separate cattle
from sheep, while keeping the rest of the network frozen.

**What actually happened during training:**
- Custom CNN trained the full 40-epoch budget until early stopping at
  **epoch 27**, restoring weights from epoch 19 (best `val_accuracy`: 86.3%).
- ResNet50 needed both phases to converge, reaching **98.65% val accuracy**
  by phase 2 epoch 11.
- MobileNetV2 and EfficientNetB0 converged unusually fast — both phases
  stopped within a handful of epochs (best weights from epoch 1 in each
  phase), landing at **96.9%** and **96.5% val accuracy** respectively.

---

## 6. Results

Test set (519 held-out images, never seen during training or validation):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **ResNet50** ★ best | 97.69% | 97.69% | 97.69% | 97.69% |
| EfficientNetB0 | 96.72% | 96.73% | 96.72% | 96.72% |
| MobileNetV2 | 96.34% | 96.42% | 96.34% | 96.34% |
| Custom CNN | 82.27% | 83.10% | 82.27% | 82.19% |

### Is the gap real, or just noise?

A **McNemar's test** compared the custom CNN against ResNet50 (the best
transfer model) on the same test images:

| | CNN correct | CNN wrong |
|---|---|---|
| **Transfer correct** | 422 | 85 |
| **Transfer wrong** | 5 | 7 |

McNemar statistic **69.34**, p-value **< 0.0001** — the difference is highly
statistically significant, not a fluke of this particular test split.

ResNet50 is the most accurate but also the heaviest to serve. If cold-start
latency matters on a free hosting tier, MobileNetV2 is a reasonable
trade-off — 96.3% vs. 97.7% accuracy for a much smaller, faster container.

---

## 7. Error Analysis

ResNet50 (the deployed default) misclassifies **12 of 519 test images
(2.3%)**. Manually inspecting those errors surfaces two recurring patterns:

- **White / pale, fluffy-coated cattle** (especially calves) get predicted
  as sheep — a close-up of a fluffy white calf face was misclassified with
  100% confidence as sheep, and a pen of woolly-looking white cows scored
  0.54 toward sheep. The coat texture visually overlaps with wool.
- **Dark-faced, patterned, or horned sheep** (and a few goat-like animals
  that survived the automated cleaning pass) get predicted as cow — a group
  of black-and-white horned sheep was misclassified as cow at 0.98
  confidence.
- A smaller number of errors come from **group shots and small/distant
  animals** — e.g. a person herding a flock in the far background of the
  frame, where the animals occupy a small fraction of the image.

**Grad-CAM** overlays confirm the model is basing its decision on the
animal's **body and coat texture**, not background context (grass, fences,
sky) — a good sign that it learned a legitimate visual signal rather than a
shortcut. Comparing the two models qualitatively: ResNet50's activation
heatmaps are tightly focused on the animal's torso, while the custom CNN's
are much more diffuse and weaker — consistent with its lower accuracy and
suggesting it hasn't learned as clean a notion of "what part of the image
matters."

**Acknowledged limitations from the training process itself:**
- Dataset cleaning is conservative — it keeps borderline cases rather than
  aggressively filtering, so some mislabelled or ambiguous images likely
  remain.
- Fine-tuning only unfreezes the top 30 backbone layers; full fine-tuning
  with a learning-rate warm-up could push accuracy further.
- No test-time augmentation (TTA) was used — ensembling predictions over
  augmented views typically adds another ~0.5–1 point of accuracy.

---

## 8. Demo

End-to-end app around the four trained models: pick a model, upload a photo,
get a prediction with confidence, see the Grad-CAM overlay explaining the
decision, browse prediction history, and view the live model-comparison
dashboard above, rendered directly from the app's `/models` endpoint.

| Feature | Where |
|---|---|
| Model selector (ResNet50 / MobileNetV2 / EfficientNetB0 / Custom CNN) | Classify tab |
| Prediction + confidence + per-class probabilities | Classify tab |
| Grad-CAM explanation, generated on demand ("why this prediction?") | Classify tab |
| Prediction history (SQLite-backed, persists across restarts) | History tab |
| Live analytics dashboard (accuracy / precision / recall / F1 per model) | Dashboard tab |

### 🎥 Demo Video

https://github.com/user-attachments/assets/58cb07ff-fd61-4a29-b345-7092e9d81535


---

## 9. Installation

**Tech stack:** Python 3.11, FastAPI, Pydantic v2, TensorFlow (CPU), Pillow,
SQLite (stdlib), pytest · React 18, Vite, TypeScript (strict), Tailwind CSS,
Recharts.

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
   results above.

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

### Testing

```bash
cd backend && pytest -v      # 25/25 — TF-free, uses fakes for the model registry
cd frontend && npm run build # type-checks (tsc -b) then builds to dist/
```

### Deployment

Three parts: push code to GitHub, host model weights on Hugging Face Hub
(too large for git), deploy the backend, deploy the frontend.

1. **Push to GitHub** — `.keras` files are gitignored (`resnet50.keras`
   alone is ~200MB, over GitHub's 100MB limit).
2. **Host weights on [Hugging Face Hub](https://huggingface.co/new)** —
   create a free Model repo, upload the 4 `.keras` files + `metadata.json`.
3. **Backend** (Render / Railway / Fly.io) — root directory `backend`,
   Docker environment, set:
   ```
   APP_HF_REPO_ID=AzizaErgasheva/cow-sheep-classifier-models
   APP_CORS_ALLOW_ORIGINS=https://https://cattle-sheep-app.vercel.app/
   ```
   `APP_CORS_ALLOW_ORIGINS` is a **plain comma-separated string** (not a
   JSON array — brackets/quotes will break it). `entrypoint.sh` downloads
   any missing model from HF Hub before starting `uvicorn`; set `HF_TOKEN`
   too if the HF repo is private.
4. **Frontend** (Vercel) — root directory `frontend`, build command
   `npm run build`, build-time env var `VITE_API_URL=https://cattle-sheep-app.onrender.com`.
5. **Close the loop** — update the backend's `APP_CORS_ALLOW_ORIGINS` to the
   real frontend URL and redeploy, or every `/predict` call will fail with a
   CORS error in the browser console.

Render's free tier sleeps after inactivity (~30–60s cold start) and has an
ephemeral filesystem, so every restart re-downloads all ~260MB of models —
a persistent disk add-on avoids that if it becomes a problem.

---

## 10. Future Improvements

- **Manual audit pass** on the cleaned dataset — the automated MobileNetV2
  filter is conservative and likely keeps some borderline/mislabelled images.
- **Full backbone fine-tuning** with a learning-rate warm-up, instead of only
  unfreezing the top 30 layers.
- **Test-time augmentation (TTA)** — ensembling predictions over augmented
  views, ~0.5–1pp expected gain.
- **Multi-class extension** beyond cow/sheep (goats, horses, etc.).
- **Active learning loop** on misclassified images surfaced by the app's own
  prediction history.
- **k-fold cross-validation** for a more robust estimate than a single split.
- **Frontend component tests** (Vitest + React Testing Library).
