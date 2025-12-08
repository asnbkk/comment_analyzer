# Веб-приложение для анализа комментариев

Веб-приложение для анализа комментариев из различных социальных сетей.

## Установка

```bash
pip install -r requirements.txt
```

## Настройка переменных окружения

Создайте файл `.env` в корне проекта со следующими переменными:

```bash
# ScrapeCreators API Key (для fetch_youtube_comments.py)
SCRAPECREATORS_API_KEY=your_api_key_here

# YouTube Data API Key (для youtube_api_comments.py)
YOUTUBE_API_KEY=your_api_key_here

# OpenAI API Key (для classification_model.ipynb)
OPENAI_API_KEY=your_api_key_here
```

**Важно:** Файл `.env` не должен попадать в git репозиторий (уже добавлен в .gitignore).

## Запуск

```bash
python3 app.py
```

Приложение будет доступно по адресу: http://localhost:5000

## Структура проекта

```
.
├── app.py                 # Основное Flask приложение
├── templates/            # HTML шаблоны
│   ├── index.html        # Главная страница
│   └── youtube.html      # Страница YouTube
├── static/               # Статические файлы
│   ├── css/
│   │   └── style.css     # Стили
│   └── js/
│       └── main.js       # JavaScript
├── yt_admin.db          # База данных SQLite
└── requirements.txt      # Зависимости Python
```

## Функционал

- **Главная страница** (`/`) - общая статистика по всем платформам
- **YouTube** (`/youtube`) - детальная статистика по YouTube
- **API** (`/api/stats`) - JSON API для получения статистики

## Перенос базы данных на другой компьютер

### Экспорт базы данных

Для экспорта базы данных используйте скрипт `export_db.py`:

```bash
# Экспорт в SQL формат (рекомендуется)
python export_db.py --format sql --output backup.sql

# Экспорт в JSON формат
python export_db.py --format json --output backup.json

# Автоматическое имя файла с датой/временем
python export_db.py --format sql
```

### Импорт базы данных

На другом компьютере используйте скрипт `import_db.py`:

```bash
# Импорт из SQL файла
python import_db.py --format sql --input backup.sql --output yt_admin.db

# Импорт из JSON файла
python import_db.py --format json --input backup.json --output yt_admin.db
```

### Альтернативный способ (простое копирование)

Если база данных не очень большая, можно просто скопировать файл `yt_admin.db`:

```bash
# На исходном компьютере
cp yt_admin.db /path/to/usb/yt_admin.db

# На новом компьютере
cp /path/to/usb/yt_admin.db yt_admin.db
```

**Примечание:** SQL формат предпочтительнее, так как сохраняет структуру таблиц и индексы.

## Социальные сети

- ✅ **YouTube** - активно
- ⏳ **Instagram** - скоро
- ⏳ **Telegram** - скоро
- ⏳ **VKontakte** - скоро

