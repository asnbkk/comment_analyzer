#!/usr/bin/env python3
"""
Тестирование модели на реальных данных из базы
"""

import sys
import sqlite3
from pathlib import Path
from transformers import pipeline

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

from analyze_comments_hf import (
    normalize_text,
    truncate_text,
    detect_language,
    connect_db,
)

DB_PATH = Path("yt_admin.db")
DEVICE = -1  # CPU

def test_model_on_samples():
    """Тестируем модель на выборке комментариев из базы"""
    
    print("Загрузка моделей...")
    try:
        classifier_ru = pipeline(
            "text-classification",
            model="SkolkovoInstitute/russian_toxicity_classifier",
            device=DEVICE,
            model_kwargs={"torch_dtype": "float32"},
        )
        print("✓ Русская модель загружена")
    except Exception as exc:
        print(f"✗ Ошибка загрузки русской модели: {exc}")
        return
    
    try:
        classifier_kk = pipeline(
            "text-classification",
            model="issai/rembert-sentiment-analysis-polarity-classification-kazakh",
            device=DEVICE,
            model_kwargs={"torch_dtype": "float32"},
        )
        print("✓ Казахская модель загружена")
    except Exception as exc:
        print(f"✗ Ошибка загрузки казахской модели: {exc}")
        classifier_kk = None
    
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 80)
    
    conn = connect_db()
    
    # Берем примеры из разных диапазонов токсичности
    test_cases = [
        ("Низкая токсичность (<0.3)", """
            SELECT cd.comment_id, cd.text_display, cd.text_original, ca.toxicity_score
            FROM comment_details cd
            JOIN comment_analysis ca ON cd.comment_id = ca.comment_id
            WHERE ca.toxicity_score < 0.3
            ORDER BY RANDOM()
            LIMIT 5
        """),
        ("Средняя токсичность (0.3-0.7)", """
            SELECT cd.comment_id, cd.text_display, cd.text_original, ca.toxicity_score
            FROM comment_details cd
            JOIN comment_analysis ca ON cd.comment_id = ca.comment_id
            WHERE ca.toxicity_score BETWEEN 0.3 AND 0.7
            ORDER BY RANDOM()
            LIMIT 5
        """),
        ("Высокая токсичность (>0.7)", """
            SELECT cd.comment_id, cd.text_display, cd.text_original, ca.toxicity_score
            FROM comment_details cd
            JOIN comment_analysis ca ON cd.comment_id = ca.comment_id
            WHERE ca.toxicity_score > 0.7
            ORDER BY RANDOM()
            LIMIT 5
        """),
    ]
    
    for category_name, query in test_cases:
        print(f"\n{category_name}:")
        print("-" * 80)
        
        rows = conn.execute(query).fetchall()
        
        for i, row in enumerate(rows, 1):
            comment_id = row["comment_id"]
            text_display = row["text_display"] or row["text_original"] or ""
            old_score = row["toxicity_score"]
            
            if not text_display.strip():
                continue
            
            # Определяем язык
            language = detect_language(text_display)
            
            # Нормализуем текст
            text_normalized = normalize_text(text_display)
            text = truncate_text(text_normalized, max_length=400)
            
            # Анализируем через модель
            try:
                if language == "kk" and classifier_kk:
                    result = classifier_kk(text, truncation=True, max_length=512)[0]
                    label = result.get("label", "")
                    score = result.get("score", 0.0)
                    if label.lower() in ["negative", "негативный", "негатив"]:
                        new_score = score
                    elif label.lower() in ["positive", "позитивный", "позитив"]:
                        new_score = 1.0 - score
                    else:
                        new_score = score
                else:
                    result = classifier_ru(text, truncation=True, max_length=512)[0]
                    label = result.get("label", "")
                    score = result.get("score", 0.0)
                    if label.lower() in ["toxic", "токсичный", "токсичность"]:
                        new_score = score
                    elif label.lower() in ["neutral", "non-toxic", "non_toxic", "non toxic", "safe"]:
                        new_score = 1.0 - score
                    else:
                        new_score = score
                
                diff = abs(new_score - old_score)
                status = "✓" if diff < 0.2 else "⚠" if diff < 0.4 else "✗"
                
                print(f"\n{status} Пример {i}:")
                print(f"   Язык: {language}")
                print(f"   Старый скор: {old_score:.3f}")
                print(f"   Новый скор:  {new_score:.3f}")
                print(f"   Разница:     {diff:.3f}")
                print(f"   Метка:       {label}")
                print(f"   Текст:        {text_display[:100]}{'...' if len(text_display) > 100 else ''}")
                
            except Exception as e:
                print(f"\n✗ Ошибка анализа примера {i}: {e}")
                print(f"   Текст: {text_display[:100]}")
    
    conn.close()
    print("\n" + "=" * 80)
    print("Тестирование завершено")


if __name__ == "__main__":
    test_model_on_samples()

