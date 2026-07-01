FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.runtime.txt /app/backend/requirements.runtime.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/backend/requirements.runtime.txt

RUN mkdir -p \
    /app/backend \
    /app/artifacts/models \
    /app/artifacts/scaler \
    /app/artifacts/threshold \
    /app/artifacts/feature_schema \
    /app/artifacts/sample_batch

COPY backend/app /app/backend/app

COPY artifacts/models/vae_best.pth /app/artifacts/models/vae_best.pth
COPY artifacts/models/model_config.json /app/artifacts/models/model_config.json
COPY artifacts/scaler/scaler.joblib /app/artifacts/scaler/scaler.joblib
COPY artifacts/scaler/imputation_medians.json /app/artifacts/scaler/imputation_medians.json
COPY artifacts/threshold/threshold.json /app/artifacts/threshold/threshold.json
COPY artifacts/feature_schema/feature_columns.json /app/artifacts/feature_schema/feature_columns.json
COPY artifacts/sample_batch/fixed_batch.csv /app/artifacts/sample_batch/fixed_batch.csv

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
