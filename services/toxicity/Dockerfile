FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code only
COPY service_toxicity.py /app/service_toxicity.py

# Default runtime config (can be overridden in compose)
ENV MODEL_RU_CHECKPOINT=cointegrated/rubert-tiny-toxicity \
    MODEL_KZ_PATH=/models/toxicity/kaz_roberta_toxic_kz \
    DEVICE=auto \
    BATCH_SIZE=64 \
    MAX_LENGTH=512 \
    PORT=8001

EXPOSE 8001

CMD ["sh", "-c", "uvicorn service_toxicity:app --host 0.0.0.0 --port ${PORT}"]

