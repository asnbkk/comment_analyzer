# Обучение модели токсичности для казахского языка

Скрипт для обучения модели детекции токсичности на казахском языке.

## Требования

```bash
pip install torch transformers datasets scikit-learn pandas numpy
```

## Использование

### Базовый запуск

```bash
cd training/toxicity
python train_kz_toxicity.py
```

### С переменными окружения

```bash
GOLDEN_CSV=../../golden_annotated_merged.csv \
BASE_MODEL=kz-transformers/kaz-roberta-conversational \
MODEL_OUT_DIR=../../models/toxicity/kaz_roberta_toxic_kz \
python train_kz_toxicity.py
```

## Параметры

### Переменные окружения

- `GOLDEN_CSV` - путь к файлу с размеченными данными (по умолчанию: `../../data/golden_annotated_merged.csv`)
- `BASE_MODEL` - базовая модель для fine-tuning (по умолчанию: `kz-transformers/kaz-roberta-conversational`)
- `MODEL_OUT_DIR` - директория для сохранения модели (по умолчанию: `../../models/toxicity/kaz_roberta_toxic_kz`)
- `CHECKPOINT_DIR` - директория для сохранения checkpoint'ов (по умолчанию: `../../models/toxicity/checkpoints`)

### Гиперпараметры (можно изменить в коде)

- `learning_rate`: 2e-5
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 10
- `max_length`: 128
- `early_stopping_patience`: 2

## Входные данные

Скрипт ожидает CSV файл (`data/golden_annotated_merged.csv`) со следующими колонками:

- `text_original` - текст комментария
- `is_kazakh_model` - флаг казахского языка (True/False)
- `toxic`, `obscene`, `threat`, `insult`, `hate` - бинарные метки токсичности

## Выходные данные

После обучения в `MODEL_OUT_DIR` будут сохранены:

- `config.json` - конфигурация модели
- `model.safetensors` или `pytorch_model.bin` - веса модели
- `tokenizer.json`, `vocab.json`, `merges.txt` - файлы токенайзера
- `meta.json` - метаданные обучения (метрики, конфигурация)

## Метрики

Модель оценивается по следующим метрикам:

- **Micro F1** - основной критерий для выбора лучшей модели
- **Macro F1** - средний F1 по всем классам
- **Per-label F1** - F1 для каждого типа токсичности (toxic, obscene, threat, insult, hate)

## Особенности

- Используется weighted loss для работы с несбалансированными классами
- Early stopping для предотвращения переобучения
- Автоматическое сохранение лучшей модели по micro_f1
- Сохранение checkpoint'ов для возможности возобновления обучения

## Пример вывода

```
[info] Loading data from ../../data/golden_annotated_merged.csv
[info] Filtered to Kazakh comments: 3004 rows
[info] Final dataset size: 3004
[info] Label distribution:
  toxic: 1104 positive
  obscene: 195 positive
  threat: 152 positive
  insult: 780 positive
  hate: 135 positive
[info] Split: train=2403, valid=300, test=301
[info] Loading base model: kz-transformers/kaz-roberta-conversational
[info] Class weights (pos_weight): tensor([ 1.7210, 14.4051, 18.7632,  2.8513, 21.2519])
[info] Starting training...
...
[ok] Model saved to ../../models/toxicity/kaz_roberta_toxic_kz
```

