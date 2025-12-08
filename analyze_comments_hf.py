#!/usr/bin/env python3
"""
Анализ комментариев через Hugging Face модель для определения токсичности.
Использует модель SkolkovoInstitute/russian_toxicity_classifier
"""

import sqlite3
import time
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import pipeline
from tqdm import tqdm

# Ограничение использования CPU для снижения нагрузки
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

DB_PATH = Path("yt_admin.db")
BATCH_SIZE = 64  # увеличенный размер батча для ускорения
DEVICE = -1  # -1 для CPU, 0 для GPU если доступно
BATCH_DELAY = 0.1  # минимальная пауза между батчами (для охлаждения)
PROGRESSIVE_DELAY = False  # отключаем прогрессивную задержку для скорости


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS comment_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comment_id TEXT UNIQUE,
            toxicity_score REAL,
            analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(comment_id) REFERENCES comment_details(comment_id)
        )
        """
    )
    conn.commit()


def fetch_unanalyzed_comments(conn: sqlite3.Connection, limit: int) -> List[Dict]:
    """Получить комментарии, которые ещё не анализировались."""
    rows = conn.execute(
        """
        SELECT cd.comment_id, cd.text_display, cd.text_original
        FROM comment_details cd
        LEFT JOIN comment_analysis ca ON cd.comment_id = ca.comment_id
        WHERE ca.comment_id IS NULL
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def count_unanalyzed_comments(conn: sqlite3.Connection) -> int:
    """Подсчитать количество комментариев, которые ещё не анализировались."""
    result = conn.execute(
        """
        SELECT COUNT(*) as total
        FROM comment_details cd
        LEFT JOIN comment_analysis ca ON cd.comment_id = ca.comment_id
        WHERE ca.comment_id IS NULL
        """
    ).fetchone()
    return result["total"] if result else 0


def detect_language(text: str) -> str:
    """
    Простая детекция языка на основе кириллицы и казахских специфических символов.
    Возвращает 'kk' для казахского, 'ru' для русского, 'mixed' для смешанного.
    """
    if not text.strip():
        return "unknown"

    # Казахские специфические символы: Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І
    kazakh_chars = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
    text_chars = set(text.lower())

    kazakh_count = len(text_chars & kazakh_chars)
    total_letters = len([c for c in text if c.isalpha()])

    # Если есть казахские символы и их достаточно много - считаем казахским
    if kazakh_count > 0 and total_letters > 0:
        kazakh_ratio = kazakh_count / total_letters
        if kazakh_ratio > 0.05:  # Если больше 5% казахских символов
            return "kk"

    # Проверка на русский (кириллица без казахских символов)
    cyrillic_pattern = re.compile(r"[а-яёА-ЯЁ]")
    if cyrillic_pattern.search(text) and kazakh_count == 0:
        return "ru"

    # Если есть и казахские и русские символы
    if kazakh_count > 0 and cyrillic_pattern.search(text):
        return "mixed"

    return "unknown"


def analyze_comments_batch(
    classifier_ru, comments: List[Dict]
) -> List[Tuple[str, float]]:
    """
    Анализирует батч комментариев через HF модель.
    Анализирует только русские комментарии, казахские пропускает.
    Возвращает список (comment_id, toxicity_score).
    """
    if not comments:
        return []

    # Фильтруем только русские комментарии
    ru_texts = []
    ru_indices = []

    for idx, c in enumerate(comments):
        text = c.get("text_display") or c.get("text_original") or ""
        if not text.strip():
            continue

        # Определяем язык на оригинальном тексте
        language = detect_language(text)

        # Пропускаем казахские комментарии
        if language == "kk":
            continue

        # Нормализуем текст
        text_normalized = normalize_text(text)
        text = truncate_text(text_normalized, max_length=400)

        # Добавляем только русские, смешанные или неизвестные комментарии
        ru_texts.append(text)
        ru_indices.append(idx)

    parsed = []

    # Обрабатываем русские комментарии
    if ru_texts:
        try:
            results_ru = classifier_ru(
                ru_texts,
                batch_size=BATCH_SIZE,
                truncation=True,
                max_length=512,
            )

            for i, result in enumerate(results_ru):
                idx = ru_indices[i]
                comment_id = comments[idx]["comment_id"]

                if isinstance(result, list):
                    toxic_result = None
                    for r in result:
                        if r.get("label", "").lower() in ["toxic", "токсичный"]:
                            toxic_result = r
                            break
                    if toxic_result:
                        result = toxic_result
                    else:
                        result = max(result, key=lambda x: x.get("score", 0))

                label = result.get("label", "")
                score = result.get("score", 0.0)

                label_lower = label.lower()
                if label_lower in ["toxic", "токсичный", "токсичность"]:
                    toxicity_score = score
                elif label_lower in [
                    "neutral",
                    "non-toxic",
                    "non_toxic",
                    "non toxic",
                    "safe",
                ]:
                    toxicity_score = 1.0 - score
                else:
                    toxicity_score = score

                parsed.append((comment_id, toxicity_score))
        except Exception as e:
            print(f"  ! Ошибка анализа русских комментариев: {e}")

    return parsed


def normalize_text(text: str) -> str:
    """
    Нормализует текст для анализа.
    Сохраняет регистр для лучшей детекции токсичности.
    Приводит к нижнему только если весь текст в верхнем регистре.
    """
    # Мягкая нормализация - сохраняем регистр для лучшей детекции
    # Приводим к нижнему только если весь текст в верхнем регистре
    if text.isupper() and len(text) > 5:
        text = text.lower()

    # Убираем множественные восклицательные знаки (оставляем один)
    text = re.sub(r"!+", "!", text)

    # Убираем множественные вопросительные знаки
    text = re.sub(r"\?+", "?", text)

    # Убираем множественные точки
    text = re.sub(r"\.+", ".", text)

    # Убираем множественные пробелы
    text = re.sub(r"\s+", " ", text)

    # Убираем пробелы перед знаками препинания
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


def truncate_text(text: str, max_length: int = 400) -> str:
    """
    Обрезает текст до максимальной длины.
    Примерно 400 символов соответствует ~512 токенам для русского текста.
    """
    if len(text) <= max_length:
        return text
    # Обрезаем и добавляем многоточие
    return text[: max_length - 3] + "..."


def save_analysis_results(
    conn: sqlite3.Connection, results: List[Tuple[str, float]]
) -> None:
    """Сохраняет результаты анализа в БД."""
    if not results:
        return

    conn.executemany(
        """
        INSERT INTO comment_analysis (
            comment_id, toxicity_score
        ) VALUES (?, ?)
        ON CONFLICT(comment_id) DO UPDATE SET
            toxicity_score=excluded.toxicity_score,
            analyzed_at=datetime('now')
        """,
        results,
    )
    conn.commit()


def main() -> None:
    print("Загрузка модели Hugging Face...")
    print("Русская модель токсичности: SkolkovoInstitute/russian_toxicity_classifier")
    print("Казахские комментарии будут пропущены (не анализируются)")

    try:
        classifier_ru = pipeline(
            "text-classification",
            model="SkolkovoInstitute/russian_toxicity_classifier",
            device=DEVICE,
            model_kwargs={"torch_dtype": "float32"},
        )
        print("✓ Русская модель загружена успешно")
    except Exception as exc:
        print(f"✗ Ошибка загрузки русской модели: {exc}")
        print("\nУбедитесь, что установлены зависимости:")
        print("  pip install transformers torch")
        return

    conn = connect_db()
    ensure_table(conn)

    # Подсчитываем общее количество комментариев для анализа
    total_to_analyze = count_unanalyzed_comments(conn)
    print(f"\nВсего комментариев для анализа: {total_to_analyze}")
    print(f"Настройки: batch_size={BATCH_SIZE}, задержка={BATCH_DELAY}с")

    if total_to_analyze == 0:
        print("Нет комментариев для анализа.")
        conn.close()
        return

    total_analyzed = 0
    total_skipped = 0
    batch_num = 0
    start_time = time.time()

    # Создаем прогресс-бар
    pbar = tqdm(total=total_to_analyze, desc="Анализ комментариев", unit="коммент")

    while True:
        comments = fetch_unanalyzed_comments(conn, limit=BATCH_SIZE)
        if not comments:
            break

        batch_num += 1

        # Анализируем только русские комментарии
        results = analyze_comments_batch(classifier_ru, comments)

        # Подсчитываем пропущенные (казахские) комментарии
        skipped_in_batch = len(comments) - len(results)
        total_skipped += skipped_in_batch

        if results:
            save_analysis_results(conn, results)
            total_analyzed += len(results)

            # Обновляем прогресс-бар
            pbar.update(len(comments))  # Обновляем на общее количество обработанных

            elapsed = time.time() - start_time
            rate = total_analyzed / elapsed if elapsed > 0 else 0
            pbar.set_postfix(
                {
                    "проанализировано": total_analyzed,
                    "пропущено": total_skipped,
                    "скорость": f"{rate:.1f} коммент/сек",
                    "батч": batch_num,
                }
            )

        # Пауза между батчами для охлаждения (минимальная)
        if PROGRESSIVE_DELAY:
            delay_multiplier = 1 + (batch_num // 50) * 0.5
            current_delay = BATCH_DELAY * delay_multiplier
        else:
            current_delay = BATCH_DELAY

        # Небольшая пауза только каждые 10 батчей для снижения нагрузки
        if batch_num % 10 == 0:
            time.sleep(current_delay)

    pbar.close()
    conn.close()
    elapsed_total = time.time() - start_time
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"Готово! Проанализировано {total_analyzed} русских комментариев")
    print(f"Пропущено {total_skipped} казахских комментариев")
    print(f"Время выполнения: {elapsed_total/60:.1f} минут")
    print(separator)


if __name__ == "__main__":
    main()
