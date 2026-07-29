# Deploying

Three parts: push code to GitHub, host your model weights on Hugging Face
Hub (they're too big for git), deploy the backend, deploy the frontend.

## 1. Push to GitHub

```powershell
cd cattle-sheep-app
git init
git add .
git status   # confirm no .keras files, node_modules, or .venv are listed
git commit -m "Initial commit: cow vs sheep classifier full-stack app"
```

Create an empty repo on github.com (skip the README/gitignore options --
you already have them), then:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

## 2. Host your models on Hugging Face Hub

Your `.keras` files are gitignored on purpose -- `resnet50.keras` alone is
~200MB, over GitHub's 100MB hard file limit.

1. Rename your local files if you haven't, so they match what the backend
   expects (`backend/models/`):
   ```
   resnet50.keras
   mobilenet.keras
   efficientnet.keras
   custom_cnn.keras
   metadata.json
   ```
2. Go to https://huggingface.co/new (free account), create a **Model** repo,
   e.g. `your-username/cow-sheep-classifier-models`.
3. Upload the 5 files above via the repo's "Files" tab → "Add file" → "Upload
   files" (drag and drop works fine, including the 200MB one).

## 3. Deploy the backend

Any Docker-friendly host works (Render, Railway, Fly.io). Render's free tier
is the simplest to start with:

1. On [render.com](https://render.com) → New → Web Service → connect your
   GitHub repo.
2. **Root directory:** `backend`
3. **Environment:** Docker (it'll auto-detect the `Dockerfile`)
4. **Environment variables:**
   ```
   APP_HF_REPO_ID=your-username/cow-sheep-classifier-models
   APP_CORS_ALLOW_ORIGINS=["https://your-frontend-domain.vercel.app"]
   ```
   (You'll fill in the real frontend URL after step 4 -- Render lets you
   edit env vars and redeploy any time.)
5. Deploy. First boot will be slow (~200MB model download inside
   `entrypoint.sh`) -- check the deploy logs for `[download_models]` lines
   to confirm it's fetching correctly.
6. Once live, note your backend URL, e.g. `https://cow-sheep-api.onrender.com`.
   Confirm it works: visit `https://your-backend-url/docs`.

## 4. Deploy the frontend

[Vercel](https://vercel.com) is the simplest for a Vite app:

1. New Project → import the same GitHub repo.
2. **Root directory:** `frontend`
3. **Build command:** `npm run build` (Vercel usually auto-detects this)
4. **Environment variable:**
   ```
   VITE_API_URL=https://your-backend-url.onrender.com
   ```
   Set this as a **build-time** variable -- Vite bakes it in at build, not
   runtime, so it must be present before/during the build step, not just at
   deploy.
5. Deploy. You'll get a URL like `https://cow-sheep-app.vercel.app`.

## 5. Close the loop: fix CORS

Go back to your Render backend's environment variables and update:
```
APP_CORS_ALLOW_ORIGINS=["https://cow-sheep-app.vercel.app"]
```
Redeploy the backend. Without this, the frontend will load but every
`/predict` call will fail with a CORS error in the browser console.

## Sanity check

Open your live frontend URL, upload a cow or sheep photo, confirm a
prediction comes back. Check the History and Dashboard tabs load too.

## Notes on free-tier limits

- Render's free web services **sleep after inactivity** and take ~30-60s to
  wake on the next request -- the first prediction after idle time will be
  slow, not broken. This is expected.
- Every restart re-runs `entrypoint.sh`, which re-downloads any model not
  already on disk. Render's free tier has an ephemeral filesystem, so this
  means every restart re-downloads all ~260MB. If that becomes a problem
  (slow cold starts, HF Hub rate limits), look into Render's persistent disk
  add-on so downloaded models survive restarts.
