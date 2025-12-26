# Веб-приложение для анализа комментариев

Веб-приложение для анализа комментариев из различных социальных сетей.

## Установка

```bash
pip install -r requirements.txt
```

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

## Социальные сети

- ✅ **YouTube** - активно
- ⏳ **Instagram** - скоро
- ⏳ **Telegram** - скоро
- ⏳ **VKontakte** - скоро

