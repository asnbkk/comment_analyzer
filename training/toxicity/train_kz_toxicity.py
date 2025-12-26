#!/usr/bin/env python3
"""
Обучение модели токсичности для казахского языка.
Использует golden_annotated_merged.csv как источник размеченных данных.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrainConfig:
    # Input data
    golden_csv: Path = Path("../../data/golden_annotated_merged.csv")
    
    # Model
    base_model: str = "kz-transformers/kaz-roberta-conversational"
    
    # Training
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.5  # от test части
    
    # Hyperparameters
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    max_length: int = 128
    early_stopping_patience: int = 2
    
    # Output
    model_out_dir: Path = Path("../../models/toxicity/kaz_roberta_toxic_kz")
    checkpoint_dir: Path = Path("../../models/toxicity/checkpoints")
    
    # Labels
    label_cols: List[str] = None
    
    def __post_init__(self):
        if self.label_cols is None:
            object.__setattr__(self, 'label_cols', ["toxic", "obscene", "threat", "insult", "hate"])


def resolve_cfg() -> TrainConfig:
    """Разрешает пути относительно этого файла"""
    base_dir = Path(__file__).resolve().parent  # training/toxicity
    
    golden_csv = Path(
        os.getenv("GOLDEN_CSV", str(base_dir / "../../data/golden_annotated_merged.csv"))
    ).resolve()
    
    model_out_dir = Path(
        os.getenv("MODEL_OUT_DIR", str(base_dir / "../../models/toxicity/kaz_roberta_toxic_kz"))
    ).resolve()
    
    checkpoint_dir = Path(
        os.getenv("CHECKPOINT_DIR", str(base_dir / "../../models/toxicity/checkpoints"))
    ).resolve()
    
    base_model = os.getenv("BASE_MODEL", "kz-transformers/kaz-roberta-conversational")
    
    return TrainConfig(
        golden_csv=golden_csv,
        base_model=base_model,
        model_out_dir=model_out_dir,
        checkpoint_dir=checkpoint_dir,
    )


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────
def load_data(cfg: TrainConfig) -> pd.DataFrame:
    """Загружает и фильтрует данные"""
    if not cfg.golden_csv.exists():
        raise FileNotFoundError(f"Golden CSV not found: {cfg.golden_csv}")
    
    print(f"[info] Loading data from {cfg.golden_csv}")
    df = pd.read_csv(cfg.golden_csv)
    
    # Оставляем только казахские комментарии
    if "is_kazakh_model" in df.columns:
        df = df[df["is_kazakh_model"] == True].copy()
        print(f"[info] Filtered to Kazakh comments: {len(df)} rows")
    else:
        print(f"[warning] Column 'is_kazakh_model' not found, using all data")
    
    # Проверяем наличие колонок с лейблами
    missing_cols = [c for c in cfg.label_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing label columns: {missing_cols}")
    
    # Убедимся, что таргеты - int
    for c in cfg.label_cols:
        df[c] = df[c].astype(int)
    
    # Проверяем наличие текста
    if "text_original" not in df.columns:
        raise ValueError("Column 'text_original' not found")
    
    # Убираем пустые тексты
    df = df[df["text_original"].astype(str).str.strip().ne("")].copy()
    
    print(f"[info] Final dataset size: {len(df)}")
    print(f"[info] Label distribution:")
    for col in cfg.label_cols:
        print(f"  {col}: {df[col].sum()} positive")
    
    return df


def split_data(df: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделяет данные на train/val/test"""
    # Сначала train/test
    train_df, temp_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=df["is_toxic"].astype(int) if "is_toxic" in df.columns else None,
    )
    
    # Затем val/test из temp
    if "is_toxic" in temp_df.columns:
        stratify = temp_df["is_toxic"].astype(int)
    else:
        stratify = None
    
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=stratify,
    )
    
    print(f"[info] Split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    
    return train_df, valid_df, test_df


def prepare_datasets(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    cfg: TrainConfig,
) -> tuple[Dataset, Dataset, Dataset]:
    """Подготавливает датасеты для обучения"""
    
    def tokenize_batch(batch):
        return tokenizer(
            batch["text_original"],
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )
    
    def add_labels(batch):
        """Добавляет labels в формате [batch_size, num_labels]"""
        labels = np.vstack([batch[c] for c in cfg.label_cols]).T.astype("float32")
        return {"labels": labels}
    
    # Создаем Dataset
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_ds = Dataset.from_pandas(valid_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
    
    # Токенизация
    train_ds_tok = train_ds.map(tokenize_batch, batched=True)
    valid_ds_tok = valid_ds.map(tokenize_batch, batched=True)
    test_ds_tok = test_ds.map(tokenize_batch, batched=True)
    
    # Добавляем labels
    train_ds_tok = train_ds_tok.map(add_labels, batched=True)
    valid_ds_tok = valid_ds_tok.map(add_labels, batched=True)
    test_ds_tok = test_ds_tok.map(add_labels, batched=True)
    
    # Устанавливаем формат
    cols = ["input_ids", "attention_mask", "labels"]
    train_ds_tok.set_format(type="torch", columns=cols)
    valid_ds_tok.set_format(type="torch", columns=cols)
    test_ds_tok.set_format(type="torch", columns=cols)
    
    return train_ds_tok, valid_ds_tok, test_ds_tok


# ─────────────────────────────────────────────────────────────
# Model & Training
# ─────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """Trainer с weighted loss для несбалансированных классов"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        
        if return_outputs:
            return loss, outputs
        return loss


def compute_metrics(eval_pred, label_cols: List[str]):
    """Вычисляет метрики для multi-label классификации"""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    
    # Порог 0.5
    y_pred = (probs >= 0.5).astype(int)
    y_true = labels
    
    # Micro/macro метрики
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Per-label F1
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        "micro_f1": micro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
    }
    
    # Добавляем F1 по каждому лейблу
    for i, name in enumerate(label_cols):
        metrics[f"f1_{name}"] = per_label_f1[i]
    
    return metrics


def train_model(
    train_ds: Dataset,
    valid_ds: Dataset,
    test_ds: Dataset,
    cfg: TrainConfig,
) -> tuple[Trainer, Dict]:
    """Обучает модель"""
    
    print(f"[info] Loading base model: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(cfg.label_cols),
        problem_type="multi_label_classification",
    )
    
    # Вычисляем веса классов
    # Используем train_ds для вычисления весов
    train_df = train_ds.to_pandas()
    label_counts = train_df[cfg.label_cols].sum()
    neg_counts = len(train_df) - label_counts
    pos_weight = (neg_counts / label_counts).values
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    
    print(f"[info] Class weights (pos_weight): {pos_weight}")
    
    # Training arguments
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(cfg.checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        fp16=torch.cuda.is_available(),
        save_total_limit=3,
        seed=cfg.seed,
    )
    
    # Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, cfg.label_cols),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )
    
    trainer.pos_weight = pos_weight
    
    # Обучение
    print("[info] Starting training...")
    train_result = trainer.train()
    
    # Оценка на тесте
    print("[info] Evaluating on test set...")
    test_metrics = trainer.evaluate(test_ds)
    print(f"[info] Test metrics: {test_metrics}")
    
    return trainer, test_metrics


# ─────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────
def save_model(trainer: Trainer, cfg: TrainConfig, test_metrics: Dict) -> None:
    """Сохраняет модель и метаданные"""
    cfg.model_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[info] Saving model to {cfg.model_out_dir}")
    
    # Сохраняем модель и токенайзер
    trainer.save_model(str(cfg.model_out_dir))
    trainer.tokenizer.save_pretrained(str(cfg.model_out_dir))
    
    # Сохраняем метаданные
    meta = {
        "model_name": "kaz_roberta_toxic_kz",
        "base_model": cfg.base_model,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": cfg.label_cols,
        "num_labels": len(cfg.label_cols),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "training_config": {
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.per_device_train_batch_size,
            "num_epochs": cfg.num_train_epochs,
            "max_length": cfg.max_length,
            "seed": cfg.seed,
        },
        "data_file": str(cfg.golden_csv),
    }
    
    meta_path = cfg.model_out_dir / "meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    print(f"[ok] Model saved to {cfg.model_out_dir}")
    print(f"[ok] Metadata saved to {meta_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    cfg = resolve_cfg()
    
    print("=" * 60)
    print("Kazakh Toxicity Model Training")
    print("=" * 60)
    print(f"[info] golden_csv:  {cfg.golden_csv}")
    print(f"[info] base_model:   {cfg.base_model}")
    print(f"[info] model_out:    {cfg.model_out_dir}")
    print(f"[info] checkpoint:   {cfg.checkpoint_dir}")
    print()
    
    # Загрузка данных
    df = load_data(cfg)
    train_df, valid_df, test_df = split_data(df, cfg)
    
    # Загрузка модели и токенайзера
    print(f"[info] Loading tokenizer from {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    
    # Подготовка датасетов
    train_ds, valid_ds, test_ds = prepare_datasets(
        train_df, valid_df, test_df, tokenizer, cfg
    )
    
    # Обучение
    trainer, test_metrics = train_model(train_ds, valid_ds, test_ds, cfg)
    
    # Сохранение
    save_model(trainer, cfg, test_metrics)
    
    print()
    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

