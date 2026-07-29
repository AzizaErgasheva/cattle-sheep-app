"""Downloads trained .keras models + metadata.json from a Hugging Face Hub
repo into backend/models/, if they aren't already present locally.

Why this exists: the models are too large (resnet50.keras is ~200MB) for a
plain git push (GitHub's hard limit is 100MB/file), and Git LFS's free
bandwidth tier gets exhausted fast if a host re-clones on every deploy.
Hosting the weights on HF Hub and pulling them at container startup keeps
the git repo small and avoids that entirely.

Safe to run every startup: if a file already exists locally, it's skipped.
If APP_HF_REPO_ID isn't set, this script does nothing -- local dev with
models already sitting in backend/models/ (e.g. via docker-compose volume
mount) works exactly as before, unaffected.
"""
import os
import sys
from pathlib import Path

MODEL_FILENAMES = [
    "resnet50.keras",
    "mobilenet.keras",
    "efficientnet.keras",
    "custom_cnn.keras",
    "metadata.json",
]


def main() -> None:
    repo_id = os.environ.get("APP_HF_REPO_ID")
    if not repo_id:
        print("[download_models] APP_HF_REPO_ID not set -- skipping download, using local models/ as-is.")
        return

    models_dir = Path(os.environ.get("APP_MODELS_DIR", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    token = os.environ.get("HF_TOKEN")  # only needed if the HF repo is private

    downloaded_any = False
    for filename in MODEL_FILENAMES:
        target = models_dir / filename
        if target.exists():
            print(f"[download_models] {filename} already present, skipping.")
            continue
        try:
            print(f"[download_models] Fetching {filename} from {repo_id} ...")
            cached_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
            target.write_bytes(Path(cached_path).read_bytes())
            downloaded_any = True
            print(f"[download_models] Saved {filename} -> {target}")
        except EntryNotFoundError:
            # Not every deployment needs all four models -- missing files are
            # fine, the registry just won't offer that architecture.
            print(f"[download_models] {filename} not found in {repo_id}, skipping (optional).")
        except Exception as exc:  # noqa: BLE001 -- surface any failure clearly, don't crash silently
            print(f"[download_models] ERROR fetching {filename}: {exc}", file=sys.stderr)

    if not downloaded_any:
        print("[download_models] Nothing new to download.")


if __name__ == "__main__":
    main()
