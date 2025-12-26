# training/lang_detect/train_langid.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrainConfig:
    # Input data (can be overridden by env)
    train_csv: Path
    valid_csv: Path
    unseen_csv: Path

    id_col: str = "comment_id"
    text_col: str = "text_original"

    seed: int = 42
    max_per_class: int = 50_000
    test_size: float = 0.2

    # Heuristic
    kz_ratio_threshold: float = 0.02

    # TF-IDF
    ngram_range: Tuple[int, int] = (3, 5)
    min_df: int = 5
    max_features: int = 200_000

    # Classifier
    max_iter: int = 1000
    n_jobs: int = -1
    class_weight: str = "balanced"

    # Output directory (models are artifacts, not code)
    model_out_dir: Path = Path("../../models/lang_detect")

    @property
    def model_path(self) -> Path:
        return self.model_out_dir / "langid_ru_kk_model.pkl"

    @property
    def report_path(self) -> Path:
        return self.model_out_dir / "langid_ru_kk_report.txt"

    @property
    def meta_path(self) -> Path:
        return self.model_out_dir / "meta.json"


def resolve_cfg() -> TrainConfig:
    """
    Defaults are relative to this file:
      training/lang_detect/train_langid.py
    so ../../data -> project_root/data
    and ../../models -> project_root/models
    """
    base_dir = Path(__file__).resolve().parent  # training/lang_detect

    train_csv = Path(os.getenv("TRAIN_CSV", str(base_dir / "../../data/train.csv"))).resolve()
    valid_csv = Path(os.getenv("VALID_CSV", str(base_dir / "../../data/validation.csv"))).resolve()
    unseen_csv = Path(os.getenv("UNSEEN_CSV", str(base_dir / "../../data/unseen.csv"))).resolve()

    model_out_dir = Path(os.getenv("MODEL_OUT_DIR", str(base_dir / "../../models/lang_detect"))).resolve()

    return TrainConfig(
        train_csv=train_csv,
        valid_csv=valid_csv,
        unseen_csv=unseen_csv,
        model_out_dir=model_out_dir,
    )


# ─────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────
KZ_LETTERS = set("әғқңөұүіӘҒҚҢӨҰҮІ")

KZ_TO_RU_MAP = str.maketrans(
    {
        "ә": "а",
        "ғ": "г",
        "қ": "к",
        "ң": "н",
        "ө": "о",
        "ұ": "у",
        "ү": "у",
        "і": "и",
        "Ә": "А",
        "Ғ": "Г",
        "Қ": "К",
        "Ң": "Н",
        "Ө": "О",
        "Ұ": "У",
        "Ү": "У",
        "І": "И",
    }
)


def _ensure_str(x) -> str:
    if isinstance(x, str):
        return x
    return "" if x is None else str(x)


def is_kazakh_heuristic(text: str, threshold: float) -> bool:
    """
    threshold = доля казахских букв среди всех букв.
    Если >= threshold — считаем текст казахским.
    """
    text = _ensure_str(text)
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    kz_count = sum(1 for ch in letters if ch in KZ_LETTERS)
    return (kz_count / len(letters)) >= threshold


def kz_to_ru_approx(text: str) -> str:
    return _ensure_str(text).translate(KZ_TO_RU_MAP)


# ─────────────────────────────────────────────────────────────
# Data loading / preparation
# ─────────────────────────────────────────────────────────────
def load_raw_df(cfg: TrainConfig) -> pd.DataFrame:
    for p in (cfg.train_csv, cfg.valid_csv, cfg.unseen_csv):
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    seen = pd.concat(
        [pd.read_csv(cfg.train_csv), pd.read_csv(cfg.valid_csv)],
        ignore_index=True,
    )[[cfg.id_col, cfg.text_col]]

    unseen = pd.read_csv(cfg.unseen_csv)[[cfg.id_col, cfg.text_col]]

    df = pd.concat([unseen, seen], ignore_index=True)
    df = df.drop_duplicates(subset=[cfg.id_col])

    # Clean
    df[cfg.text_col] = df[cfg.text_col].map(_ensure_str)
    df = df[df[cfg.text_col].str.strip().ne("")].copy()

    return df


def build_labeled_dataset(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    is_kz = df[cfg.text_col].apply(lambda t: is_kazakh_heuristic(t, cfg.kz_ratio_threshold))
    df_kz = df[is_kz].copy()
    df_ru = df[~is_kz].copy()

    n_kz = min(len(df_kz), cfg.max_per_class)
    n_ru = min(len(df_ru), cfg.max_per_class)

    if n_kz > 0:
        df_kz = df_kz.sample(n_kz, random_state=cfg.seed)
    if n_ru > 0:
        df_ru = df_ru.sample(n_ru, random_state=cfg.seed)

    df_kz["lang"] = "kk"
    df_ru["lang"] = "ru"

    df_lang = pd.concat([df_kz, df_ru], ignore_index=True)

    # Augment: kz text with RU-like letters but keep label kk
    if len(df_kz) > 0:
        df_kz_aug = df_kz.copy()
        df_kz_aug[cfg.text_col] = df_kz_aug[cfg.text_col].apply(kz_to_ru_approx)
        df_lang = pd.concat([df_lang, df_kz_aug], ignore_index=True)

    df_lang = df_lang.dropna(subset=[cfg.text_col]).copy()
    df_lang[cfg.text_col] = df_lang[cfg.text_col].map(_ensure_str)
    df_lang = df_lang[df_lang[cfg.text_col].str.strip().ne("")].copy()

    # Shuffle
    df_lang = df_lang.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    return df_lang


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
def build_pipeline(cfg: TrainConfig) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=cfg.ngram_range,
                    min_df=cfg.min_df,
                    max_features=cfg.max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=cfg.max_iter,
                    n_jobs=cfg.n_jobs,
                    class_weight=cfg.class_weight,
                ),
            ),
        ]
    )


def train_and_evaluate(df_lang: pd.DataFrame, cfg: TrainConfig) -> tuple[Pipeline, str, dict]:
    X = df_lang[cfg.text_col].astype(str)
    y = df_lang["lang"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )

    pipeline = build_pipeline(cfg)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)

    counts = df_lang["lang"].value_counts().to_dict()
    info = {
        "dataset_size": int(len(df_lang)),
        "counts": {k: int(v) for k, v in counts.items()},
        "test_size": cfg.test_size,
        "seed": cfg.seed,
    }

    return pipeline, report, info


# ─────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────
def save_artifacts(pipeline: Pipeline, report: str, meta: dict, cfg: TrainConfig) -> None:
    cfg.model_out_dir.mkdir(parents=True, exist_ok=True)

    # model
    joblib.dump(pipeline, cfg.model_path)

    # report
    cfg.report_path.write_text(report, encoding="utf-8")

    # meta
    meta_out = {
        "model_name": "lang_detect",
        "labels": ["ru", "kk"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": cfg.model_path.name,
        "report_file": cfg.report_path.name,
        "data_files": {
            "train_csv": str(cfg.train_csv),
            "valid_csv": str(cfg.valid_csv),
            "unseen_csv": str(cfg.unseen_csv),
        },
        "params": {
            "kz_ratio_threshold": cfg.kz_ratio_threshold,
            "max_per_class": cfg.max_per_class,
            "tfidf": {
                "ngram_range": cfg.ngram_range,
                "min_df": cfg.min_df,
                "max_features": cfg.max_features,
            },
            "clf": {
                "max_iter": cfg.max_iter,
                "n_jobs": cfg.n_jobs,
                "class_weight": cfg.class_weight,
            },
        },
        "run_info": meta,
    }
    cfg.meta_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    cfg = resolve_cfg()

    print(f"[info] train_csv:  {cfg.train_csv}")
    print(f"[info] valid_csv:  {cfg.valid_csv}")
    print(f"[info] unseen_csv: {cfg.unseen_csv}")
    print(f"[info] out_dir:    {cfg.model_out_dir}")

    df = load_raw_df(cfg)
    df_lang = build_labeled_dataset(df, cfg)

    print(f"[info] labeled dataset size={len(df_lang)}")
    print(f"[info] class counts:\n{df_lang['lang'].value_counts()}")

    pipeline, report, meta = train_and_evaluate(df_lang, cfg)

    print(report)
    save_artifacts(pipeline, report, meta, cfg)

    print(f"[ok] saved model  -> {cfg.model_path}")
    print(f"[ok] saved report -> {cfg.report_path}")
    print(f"[ok] saved meta   -> {cfg.meta_path}")


if __name__ == "__main__":
    main()
