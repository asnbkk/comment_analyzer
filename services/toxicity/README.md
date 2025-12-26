# Toxicity Analysis Service (RU/KK)

Единый сервис для анализа токсичности русских и казахских текстов.

## Модели

- **Русский язык**: `cointegrated/rubert-tiny-toxicity` (загружается из HuggingFace)
- **Казахский язык**: локальная модель из `/models/kaz_roberta_toxic_kz`

## Возможности

- Анализ токсичности одного текста
- Батчевая обработка нескольких текстов
- Автоматическое определение языка или явное указание
- Детальная оценка по 5 аспектам:
  - `toxic` - общая агрессивность
  - `obscene` - нецензурная лексика
  - `threat` - угрозы
  - `insult` - оскорбления
  - `hate` - ненависть к группам
- Агрегированный скор токсичности (разные формулы для RU и KK)

## Запуск

### Через Docker Compose

```bash
docker-compose up toxicity
```

### Локально

```bash
cd services/toxicity
pip install -r requirements.txt
uvicorn service_toxicity:app --host 0.0.0.0 --port 8001
```

## API

### Health Check

```bash
curl http://localhost:8001/health
```

**Ответ:**
```json
{
  "status": "ok",
  "models_loaded": {
    "ru": true,
    "kk": true
  },
  "model_ru_checkpoint": "cointegrated/rubert-tiny-toxicity",
  "model_kz_path": "/models/kaz_roberta_toxic_kz",
  "device": "cpu",
  "batch_size": 64,
  "max_length": 512
}
```

### Анализ одного текста

#### С автоопределением языка

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Ваш текст здесь", "lang": "auto"}'
```

#### С явным указанием языка

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Ваш текст здесь", "lang": "ru"}'
```

**Ответ:**
```json
{
  "toxicity_score": 0.85,
  "aspects": {
    "toxic": 0.9,
    "obscene": 0.7,
    "threat": 0.1,
    "insult": 0.8,
    "hate": 0.2
  },
  "lang": "ru"
}
```

### Батчевая обработка

```bash
curl -X POST http://localhost:8001/analyze_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Текст 1", "Текст 2", "Текст 3"],
    "lang": "auto"
  }'
```

**Ответ:**
```json
{
  "results": [
    {
      "toxicity_score": 0.85,
      "aspects": {
        "toxic": 0.9,
        "obscene": 0.7,
        "threat": 0.1,
        "insult": 0.8,
        "hate": 0.2
      },
      "lang": "ru"
    },
    ...
  ]
}
```

## Переменные окружения

- `MODEL_RU_CHECKPOINT` - путь к русской модели (по умолчанию: `cointegrated/rubert-tiny-toxicity`)
- `MODEL_KZ_PATH` - путь к казахской модели (по умолчанию: `/models/kaz_roberta_toxic_kz`)
- `DEVICE` - устройство для вычислений: `auto`, `cpu`, `cuda` (по умолчанию: `auto`)
- `BATCH_SIZE` - размер батча для обработки (по умолчанию: `64`)
- `MAX_LENGTH` - максимальная длина текста в токенах (по умолчанию: `512`)
- `PORT` - порт сервиса (по умолчанию: `8001`)

## Формулы агрегированного скора

### Русский язык
```
toxicity_score = 1 - P(toxic) * (1 - P(hate))
```

### Казахский язык
```
toxicity_score = 1 - ∏(1 - P_i)  для всех 5 аспектов
```

Где `P_i` - вероятности по каждому аспекту (toxic, obscene, threat, insult, hate).

## Автоопределение языка

Если `lang="auto"`, сервис использует простую эвристику:
- Если в тексте >5% казахских специфических символов (ә, ғ, қ, ң, ө, ұ, ү, һ, і) - казахский
- Иначе - русский

