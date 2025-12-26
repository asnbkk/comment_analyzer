from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ServiceConfig:
    model_ru_checkpoint: str = os.getenv(
        "MODEL_RU_CHECKPOINT", "cointegrated/rubert-tiny-toxicity"
    )
    model_kz_path: Path = Path(os.getenv("MODEL_KZ_PATH", "/models/toxicity/kaz_roberta_toxic_kz"))
    device: str = os.getenv("DEVICE", "auto")  # auto, cpu, cuda
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))
    max_length: int = int(os.getenv("MAX_LENGTH", "512"))


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _ensure_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    return "" if x is None else str(x)


def _clean_texts(texts: List[Any]) -> List[str]:
    """Очистка входных текстов: все к строке, NaN/None -> ''"""
    cleaned = []
    for t in texts:
        if t is None or (isinstance(t, float) and pd.isna(t)):
            cleaned.append("")
        else:
            cleaned.append(_ensure_str(t))
    return cleaned


# ─────────────────────────────────────────────────────────────
# API schemas
# ─────────────────────────────────────────────────────────────
class ToxicityRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for toxicity")
    lang: Optional[Literal["ru", "kk", "auto"]] = Field(
        "auto", description="Language: 'ru' for Russian, 'kk' for Kazakh, 'auto' for auto-detect"
    )


class ToxicityBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts (batch)")
    lang: Optional[Literal["ru", "kk", "auto"]] = Field(
        "auto", description="Language: 'ru' for Russian, 'kk' for Kazakh, 'auto' for auto-detect"
    )


class ToxicityAspects(BaseModel):
    toxic: float = Field(..., description="General toxicity score")
    obscene: float = Field(..., description="Obscene language score")
    threat: float = Field(..., description="Threat score")
    insult: float = Field(..., description="Insult score")
    hate: float = Field(..., description="Hate speech score")


class ToxicityResponse(BaseModel):
    toxicity_score: float = Field(..., description="Aggregated toxicity score")
    aspects: ToxicityAspects = Field(..., description="Individual aspect scores")
    lang: str = Field(..., description="Detected or specified language")


class ToxicityBatchResponse(BaseModel):
    results: List[ToxicityResponse]


# ─────────────────────────────────────────────────────────────
# Model wrappers
# ─────────────────────────────────────────────────────────────
class ToxicityAnalyzerRU:
    """Анализатор токсичности для русского языка"""

    def __init__(
        self,
        model_checkpoint: str,
        device: torch.device,
        batch_size: int,
        max_length: int,
    ):
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"[init] Loading RU model from {model_checkpoint}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint
        )
        self.model.to(device)
        self.model.eval()

        self.aspect_names = ["toxic", "obscene", "threat", "insult", "hate"]
        print(f"[init] RU model loaded on {device}")

    def _predict_batch(self, texts: List[str]) -> np.ndarray:
        """Предсказание для батча текстов. Возвращает [batch_size, 5]"""
        cleaned_texts = _clean_texts(texts)

        with torch.no_grad():
            inputs = self.tokenizer(
                cleaned_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            ).to(self.device)

            logits = self.model(**inputs).logits
            proba = torch.sigmoid(logits).cpu().numpy()

        return proba

    def _aggregate_score(self, proba: np.ndarray) -> float:
        """
        Агрегированный скор для русского языка.
        Формула: 1 - proba[:, 0] * (1 - proba[:, -1])
        где proba[:, 0] = toxic, proba[:, -1] = hate
        """
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        return float(1 - proba[0, 0] * (1 - proba[0, -1]))


class ToxicityAnalyzerKZ:
    """Анализатор токсичности для казахского языка"""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        batch_size: int,
        max_length: int,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        if not model_path.exists():
            raise FileNotFoundError(f"KZ model not found: {model_path}")

        print(f"[init] Loading KZ model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(device)
        self.model.eval()

        self.aspect_names = ["toxic", "obscene", "threat", "insult", "hate"]
        print(f"[init] KZ model loaded on {device}")

    def _predict_batch(self, texts: List[str]) -> np.ndarray:
        """Предсказание для батча текстов. Возвращает [batch_size, 5]"""
        cleaned_texts = _clean_texts(texts)

        with torch.no_grad():
            inputs = self.tokenizer(
                cleaned_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            ).to(self.device)

            logits = self.model(**inputs).logits
            proba = torch.sigmoid(logits).cpu().numpy()

        return proba

    def _aggregate_score(self, proba: np.ndarray) -> float:
        """
        Агрегированный скор для казахского языка.
        Формула: 1 - prod(1 - p_i) по всем 5 аспектам (union probability)
        """
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        return float(1.0 - np.prod(1.0 - proba[0, :]))


# ─────────────────────────────────────────────────────────────
# Unified analyzer
# ─────────────────────────────────────────────────────────────
class UnifiedToxicityAnalyzer:
    """Единый анализатор токсичности для русского и казахского языков"""

    def __init__(
        self,
        analyzer_ru: ToxicityAnalyzerRU,
        analyzer_kz: Optional[ToxicityAnalyzerKZ],
    ):
        self.analyzer_ru = analyzer_ru
        self.analyzer_kz = analyzer_kz

    def _detect_language_simple(self, text: str) -> str:
        """
        Простая эвристика для определения языка.
        Если есть казахские специфические символы - казахский, иначе русский.
        """
        kazakh_chars = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
        text_chars = set(text.lower())

        kazakh_count = len(text_chars & kazakh_chars)
        total_letters = len([c for c in text if c.isalpha()])

        if kazakh_count > 0 and total_letters > 0:
            kazakh_ratio = kazakh_count / total_letters
            if kazakh_ratio > 0.05:  # Если больше 5% казахских символов
                return "kk"

        return "ru"

    def analyze_one(
        self, text: str, lang: str = "auto"
    ) -> ToxicityResponse:
        """Анализ одного текста"""
        # Определяем язык
        if lang == "auto":
            detected_lang = self._detect_language_simple(text)
        else:
            detected_lang = lang

        # Выбираем анализатор
        if detected_lang == "kk":
            if self.analyzer_kz is None:
                raise HTTPException(
                    status_code=503, detail="Kazakh model is not loaded"
                )
            analyzer = self.analyzer_kz
        else:
            analyzer = self.analyzer_ru

        # Предсказание
        proba = analyzer._predict_batch([text])[0]  # [5]

        # Агрегированный скор
        toxicity_score = analyzer._aggregate_score(proba)

        # Аспекты
        aspects = ToxicityAspects(
            toxic=float(proba[0]),
            obscene=float(proba[1]),
            threat=float(proba[2]),
            insult=float(proba[3]),
            hate=float(proba[4]),
        )

        return ToxicityResponse(
            toxicity_score=toxicity_score,
            aspects=aspects,
            lang=detected_lang,
        )

    def analyze_batch(
        self, texts: List[str], lang: str = "auto"
    ) -> ToxicityBatchResponse:
        """Анализ батча текстов"""
        results = []

        # Если язык указан явно, используем один анализатор для всех
        if lang != "auto":
            if lang == "kk":
                if self.analyzer_kz is None:
                    raise HTTPException(
                        status_code=503, detail="Kazakh model is not loaded"
                    )
                analyzer = self.analyzer_kz
            else:
                analyzer = self.analyzer_ru

            # Обрабатываем батчами
            for i in range(0, len(texts), analyzer.batch_size):
                batch_texts = texts[i : i + analyzer.batch_size]
                proba_batch = analyzer._predict_batch(batch_texts)

                for proba in proba_batch:
                    toxicity_score = analyzer._aggregate_score(proba)
                    aspects = ToxicityAspects(
                        toxic=float(proba[0]),
                        obscene=float(proba[1]),
                        threat=float(proba[2]),
                        insult=float(proba[3]),
                        hate=float(proba[4]),
                    )
                    results.append(
                        ToxicityResponse(
                            toxicity_score=toxicity_score,
                            aspects=aspects,
                            lang=lang,
                        )
                    )
        else:
            # Автоопределение языка для каждого текста
            for text in texts:
                result = self.analyze_one(text, lang="auto")
                results.append(result)

        return ToxicityBatchResponse(results=results)


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────
cfg = ServiceConfig()
app = FastAPI(
    title="Toxicity Analysis Service (RU/KK)", version="1.0.0"
)

analyzer: Optional[UnifiedToxicityAnalyzer] = None


@app.on_event("startup")
def startup():
    global analyzer
    try:
        device = _get_device(cfg.device)

        # Загружаем русскую модель
        analyzer_ru = ToxicityAnalyzerRU(
            model_checkpoint=cfg.model_ru_checkpoint,
            device=device,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
        )

        # Загружаем казахскую модель (если доступна)
        analyzer_kz = None
        try:
            analyzer_kz = ToxicityAnalyzerKZ(
                model_path=cfg.model_kz_path,
                device=device,
                batch_size=cfg.batch_size,
                max_length=cfg.max_length,
            )
        except Exception as e:
            print(f"[startup] Warning: KZ model not loaded: {e}")
            print("[startup] Service will work only for Russian texts")

        analyzer = UnifiedToxicityAnalyzer(analyzer_ru, analyzer_kz)
        print(f"[startup] Service ready on {device}")
    except Exception as e:
        analyzer = None
        print(f"[startup] Failed to load models: {e}")
        raise


@app.get("/health")
def health():
    return {
        "status": "ok" if analyzer else "degraded",
        "models_loaded": {
            "ru": analyzer.analyzer_ru is not None if analyzer else False,
            "kk": analyzer.analyzer_kz is not None if analyzer else False,
        },
        "model_ru_checkpoint": cfg.model_ru_checkpoint,
        "model_kz_path": str(cfg.model_kz_path),
        "device": str(analyzer.analyzer_ru.device) if analyzer and analyzer.analyzer_ru else None,
        "batch_size": cfg.batch_size,
        "max_length": cfg.max_length,
    }


@app.post("/analyze", response_model=ToxicityResponse)
def analyze(req: ToxicityRequest):
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    return analyzer.analyze_one(req.text, lang=req.lang)


@app.post("/analyze_batch", response_model=ToxicityBatchResponse)
def analyze_batch(req: ToxicityBatchRequest):
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    return analyzer.analyze_batch(req.texts, lang=req.lang)

