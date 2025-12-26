from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ServiceConfig:
    model_path: Path = Path(os.getenv("MODEL_PATH", "/models/langid_ru_kk_model.pkl"))
    min_len: int = int(os.getenv("MIN_LEN", "5"))
    proba_threshold: float = float(os.getenv("PROBA_THRESHOLD", "0.60"))
    return_proba: bool = os.getenv("RETURN_PROBA", "1").lower() not in {"0", "false"}
    meta_path: Path = Path(os.getenv("MODEL_META_PATH", "/models/meta.json"))


def _ensure_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    return "" if x is None else str(x)


def _normalize_text(text: str) -> str:
    return _ensure_str(text).strip()


# ─────────────────────────────────────────────────────────────
# API schemas
# ─────────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    text: str = Field(..., description="Text to detect language for")


class DetectBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts (batch)")


class DetectResponse(BaseModel):
    lang: str
    confidence: Optional[float] = None
    proba: Optional[Dict[str, float]] = None


class DetectBatchResponse(BaseModel):
    results: List[DetectResponse]


# ─────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────
class LangDetector:
    def __init__(self, model_path: Path, meta_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.pipeline = joblib.load(model_path)

        if not hasattr(self.pipeline, "predict"):
            raise TypeError("Loaded object does not have predict()")

        self.model_path = model_path
        self.meta = self._load_meta(meta_path)

    @staticmethod
    def _load_meta(path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    @property
    def classes(self) -> List[str]:
        return list(getattr(self.pipeline, "classes_", []))

    def predict_one(
        self,
        text: str,
        *,
        min_len: int,
        proba_threshold: float,
        return_proba: bool
    ) -> DetectResponse:
        raw = _ensure_str(text)
        norm = _normalize_text(raw)

        if len(norm) < min_len:
            return DetectResponse(lang="unknown")

        label = self.pipeline.predict([norm])[0]

        proba_dict: Dict[str, float] = {}
        confidence: Optional[float] = None

        if return_proba and hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([norm])[0]
            classes = self.pipeline.classes_
            proba_dict = {str(c): float(p) for c, p in zip(classes, probs)}
            confidence = max(proba_dict.values())

            if confidence < proba_threshold:
                label = "unknown"

        return DetectResponse(
            lang=str(label),
            confidence=confidence,
            proba=proba_dict if return_proba else None,
        )

    def predict_batch(
        self,
        texts: List[str],
        *,
        min_len: int,
        proba_threshold: float,
        return_proba: bool
    ) -> DetectBatchResponse:
        return DetectBatchResponse(
            results=[
                self.predict_one(
                    t,
                    min_len=min_len,
                    proba_threshold=proba_threshold,
                    return_proba=return_proba,
                )
                for t in texts
            ]
        )


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────
cfg = ServiceConfig()
app = FastAPI(title="RU/KK Language Detection Service", version="1.0.0")

detector: Optional[LangDetector] = None


@app.on_event("startup")
def startup():
    global detector
    try:
        detector = LangDetector(cfg.model_path, cfg.meta_path)
        print(f"[startup] model loaded from {cfg.model_path}")
    except Exception as e:
        detector = None
        print(f"[startup] failed to load model: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok" if detector else "degraded",
        "model_loaded": detector is not None,
        "model_path": str(cfg.model_path),
        "classes": detector.classes if detector else [],
        "meta": detector.meta if detector else {},
        "min_len": cfg.min_len,
        "proba_threshold": cfg.proba_threshold,
    }


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    return detector.predict_one(
        req.text,
        min_len=cfg.min_len,
        proba_threshold=cfg.proba_threshold,
        return_proba=cfg.return_proba,
    )


@app.post("/detect_batch", response_model=DetectBatchResponse)
def detect_batch(req: DetectBatchRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    return detector.predict_batch(
        req.texts,
        min_len=cfg.min_len,
        proba_threshold=cfg.proba_threshold,
        return_proba=cfg.return_proba,
    )
