# syntax=docker/dockerfile:1
#
# BEFORE BUILDING IN PRODUCTION:
#   1. Pin the exact image digest:
#        docker pull python:3.11.9-slim
#        docker inspect python:3.11.9-slim --format '{{index .RepoDigests 0}}'
#      Then replace "python:3.11.9-slim" below with
#      "python:3.11.9-slim@sha256:<digest>"
#
#   2. Pin the uv version the same way using ghcr.io/astral-sh/uv:<tag>
#
#   3. Mount data/ and models/ read-only at runtime — never bake index
#      files into the image.  Example docker run:
#        docker run \
#          -v $(pwd)/data:/app/data:ro \
#          -v $(pwd)/models:/app/models:ro \
#          -e HF_HUB_OFFLINE=1 \
#          -p 8000:8000 \
#          hybrid-search-engine

FROM python:3.11.9-slim

# Prevent .pyc files and enable unbuffered stdout/stderr for clean logs.
# HF_HUB_OFFLINE=1 tells HuggingFace libraries not to attempt any downloads
# at runtime — all model files must be mounted from the host.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

# ── Install uv for fast, reproducible dependency installation ──────────────
# Using COPY --from avoids a full pip install of uv inside the image.
# Pin the tag to a specific uv release before shipping to production.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# ── Install Python dependencies ───────────────────────────────────────────
# Copy only the dependency manifest first so Docker can cache this layer.
# If pyproject.toml (and uv.lock when present) don't change, this layer
# is reused even when source code changes.
COPY pyproject.toml ./

# Install runtime deps only (no dev tools in the image).
# --no-install-project avoids copying src/ into the venv at this stage.
RUN uv sync --no-dev --no-install-project 2>/dev/null || \
    pip install --no-cache-dir -e "."

# ── Copy application source ───────────────────────────────────────────────
COPY src ./src

# ── Non-root user ─────────────────────────────────────────────────────────
# Running as root inside a container is a security risk: if the app is
# compromised, the attacker has root access to the container filesystem.
# Creating a dedicated user limits blast radius.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Healthcheck ───────────────────────────────────────────────────────────
# Docker (and Kubernetes) use this to decide whether the container is
# healthy. We hit /health/ready (not /health/live) because we want the
# container to be considered unhealthy until search services are loaded.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')" \
    || exit 1

# ── One worker ────────────────────────────────────────────────────────────
# A single worker keeps memory predictable: one FAISS index + one set of
# models in RAM.  To handle more traffic, run more *replicas* (containers)
# behind a load balancer instead of adding workers — each worker would load
# its own copy of the multi-GB index.
EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "uvicorn", \
     "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
