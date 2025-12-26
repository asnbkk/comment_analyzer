FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code only (no model in the image)
COPY service_lang_detect.py /app/service_lang_detect.py

# Default runtime config (can be overridden in compose)
ENV MODEL_PATH=/models/langid_ru_kk_model.pkl \
    MODEL_META_PATH=/models/meta.json \
    MIN_LEN=5 \
    PROBA_THRESHOLD=0.60 \
    RETURN_PROBA=1 \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn service_lang_detect:app --host 0.0.0.0 --port ${PORT}"]
