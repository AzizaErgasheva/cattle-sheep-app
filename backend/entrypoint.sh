#!/bin/sh
set -e

python scripts/download_models.py

exec uvicorn app.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
