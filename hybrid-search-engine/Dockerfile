# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.11.27 AS uv
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/model-cache \
    SENTENCE_TRANSFORMERS_HOME=/app/model-cache/sentence-transformers \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app
COPY --from=uv /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra metrics --no-install-project
COPY src ./src

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/model-cache \
    && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD /app/.venv/bin/python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4)" \
    || exit 1

EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
