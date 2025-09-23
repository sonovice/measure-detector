# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
LABEL maintainer="Simon Waloschek <waloschek@pm.me>"

WORKDIR /app

# install system deps: git, curl for uv installer
RUN apt-get update && \
    apt-get install -y git curl ca-certificates libgl1 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:${PATH}"
ENV UV_PYTHON=/opt/conda/bin/python UV_LINK_MODE=copy

# ensure Ultralytics config dir is writable to avoid warnings
ENV YOLO_CONFIG_DIR=/app/.config/Ultralytics
RUN mkdir -p /app/.config/Ultralytics

# copy project metadata and lock (if present), then install deps with uv
COPY pyproject.toml README.md LICENSE ./
# set fixed model source and warm the HF cache during build
ENV HF_REPO_ID=sonovice/measure-detector \
    HF_MODEL_FILENAME=model.pt \
    HF_REVISION=main
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/huggingface \
    uv sync --no-dev --no-install-project && \
    /app/.venv/bin/python - <<'PY'
import os
import shutil
from huggingface_hub import hf_hub_download

repo_id = os.getenv('HF_REPO_ID', 'sonovice/measure-detector')
filename = os.getenv('HF_MODEL_FILENAME', 'model.pt')
revision = os.getenv('HF_REVISION', 'main')
dst = '/app/model.pt'

path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
shutil.copyfile(path, dst)
print('Saved model to', dst)
PY

# clone YOLOv5 locally to avoid runtime hub downloads
RUN git clone --depth 1 https://github.com/ultralytics/yolov5.git /app/yolov5

# On arm64 (CPU-only), install PyTorch CPU wheels into the uv environment
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      uv pip install torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cpu ; \
    fi

# copy application code and install project into venv
COPY src/ src/
RUN /app/.venv/bin/yolo settings && \
    /app/.venv/bin/yolo settings datasets_dir=/app/data runs_dir=/app/runs update_check=False
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --no-dev
ENV HF_HUB_OFFLINE=1
# ensure venv binaries (gunicorn, uvicorn, etc.) are on PATH
ENV PATH="/app/.venv/bin:${PATH}"
COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

RUN pip freeze

# reduce warnings
ENV TORCH_CPP_LOG_LEVEL=ERROR

ENV WORKERS=1

EXPOSE 8123

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8123/health || exit 1

CMD ["/app/entrypoint.sh"]


