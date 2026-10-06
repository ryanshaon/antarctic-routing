# Antarctic Vessel Routing & Ice-Risk Forecasting System.
#
# Two images from one file:
#   docker build -t antarctic-routing .                        # API + dashboard (default, e.g. Render)
#   docker build --target worker -t antarctic-routing:worker . # CLI: ingestion, training, plan-window, bundles
#
# Neither image holds scientific datasets, model weights or credentials. The API image
# may hold the small, checksum-verified real-data bundle from deploy/bundle (see
# docs/DEPLOYMENT.md); the API verifies it at start-up and never computes it.
FROM python:3.11-slim AS python

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    ANTROUTE_CONFIG=/app/config/config.yaml \
    ANTROUTE_FIGURES=/app/docs/images

WORKDIR /app
RUN useradd --create-home --uid 1000 antroute


# Batch worker: CPU-only PyTorch plus the data-service clients. Credentials are passed
# to this image at run time only (environment or secret files), never baked in.
FROM python AS worker
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,ml]" requests cdsapi netCDF4
# Full CMEMS client (heavier): pip install ".[ingest]"
COPY config ./config
COPY docs ./docs
RUN mkdir -p /app/data /app/reports /app/models && chown -R antroute /app
USER antroute
CMD ["antroute", "--help"]


# API (default target): serves the dashboard, the synthetic planner and one verified
# real-data bundle read-only. No PyTorch, no ingestion clients, no credentials.
FROM python AS api
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"
COPY config ./config
COPY docs ./docs
COPY deploy ./deploy
RUN mkdir -p /app/bundle && chown -R antroute /app
USER antroute

# Render (and similar platforms) set PORT; 8000 otherwise.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000')).status == 200 else 1)"

# One process: jobs and voyages live in memory. ANTROUTE_ARTIFACTS_DIR selects the bundle
# (/app/deploy/bundle when it was built into the image, /app/bundle for a mounted one).
CMD ["sh", "-c", "exec antroute serve --host 0.0.0.0 --port \"${PORT:-8000}\" --config /app/config/config.yaml"]
