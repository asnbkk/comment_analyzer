#!/usr/bin/env python3
"""
Сравнение разных моделей токсичности для русского языка
"""

from transformers import pipeline

models_to_test = [
    "SkolkovoInstitute/russian_toxicity_classifier",  # Текущая модель
    "ai-forever/ruBert-base-toxicity-classifier",  # Альтернатива 1
    "cointegrated/rubert-tiny2-toxicity",  # Альтернатива 2
]

test_texts = [
    "Спасибо большое за вашу работу!",
    "Ты дебил, иди отсюда",
    "Убить всех этих ублюдков!",
    "Народ трусы потому что , и это схавают!",
    "Молодец, поддерживаю!",
]

print("=" * 80)
print("СРАВНЕНИЕ МОДЕЛЕЙ ТОКСИЧНОСТИ")
print("=" * 80)

for model_name in models_to_test:
    print(f"\n{'='*80}")
    print(f"Модель: {model_name}")
    print("=" * 80)
    
    try:
        classifier = pipeline(
            "text-classification",
            model=model_name,
            device=-1,  # CPU
        )
        
        for text in test_texts:
            try:
                result = classifier(text, truncation=True, max_length=512)[0]
                label = result.get("label", "")
                score = result.get("score", 0.0)
                
                # Определяем токсичность в зависимости от метки
                if "toxic" in label.lower() or "токсич" in label.lower():
                    toxicity = score
                elif "neutral" in label.lower() or "non-toxic" in label.lower() or "safe" in label.lower():
                    toxicity = 1.0 - score
                else:
                    toxicity = score
                
                print(f"\nТекст: {text[:60]}...")
                print(f"  Метка: {label}")
                print(f"  Скор: {score:.3f}")
                print(f"  Токсичность: {toxicity:.3f}")
            except Exception as e:
                print(f"\nОшибка при анализе текста: {e}")
        
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        print("  Модель недоступна или требует авторизации")

print("\n" + "=" * 80)





