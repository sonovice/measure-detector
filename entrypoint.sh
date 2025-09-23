#!/usr/bin/env sh
set -e

# Ensure Python can import from repo root and src/ (local dev uses src layout)
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src:${PYTHONPATH}"

WORKERS=${WORKERS:-1}

exec gunicorn \
  server:app \
  --workers "$WORKERS" \
  --worker-class uvicorn.workers.UvicornWorker \
  --preload \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --graceful-timeout 30 \
  --keep-alive 5 \
  --timeout 120 \
  --bind 0.0.0.0:8123


