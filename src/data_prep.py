"""
data_prep.py — Load, preprocess, and prepare FLEURS Georgian data for Whisper fine-tuning.

Handles:
  - Loading FLEURS ka_ge (train / validation / test)
  - Optional Common Voice Georgian loading & merging
  - Resampling to 16 kHz mono
  - Dropping clips over max_duration_s
  - Feature extraction (log-Mel spectrogram) & tokenization
  - Train/test transcript overlap check and removal
"""

import os
import unicodedata
import re
import string
from functools import partial

import yaml
import numpy as np
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load the project configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Text normalization (shared across training, eval, and demo)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize_text(text: str, cfg: dict | None = None) -> str:
    """Normalize a transcription string.

    Steps (all configurable via config.yaml → evaluation.normalization):
      1. Unicode NFC
      2. Lowercase
      3. Strip punctuation
      4. Collapse whitespace
    """
    if cfg is None:
        cfg = {
            "unicode_nfc": True,
            "lowercase": True,
            "strip_punctuation": True,
            "collapse_whitespace": True,
        }

    if cfg.get("unicode_nfc", True):
        text = unicodedata.normalize("NFC", text)
    if cfg.get("lowercase", True):
        text = text.lower()
    if cfg.get("strip_punctuation", True):
        text = _PUNCT_RE.sub("", text)
    if cfg.get("collapse_whitespace", True):
        text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_fleurs(subset: str = "ka_ge", split: str = "train") -> "Dataset":
    """Load a FLEURS split."""
    ds = load_dataset("google/fleurs", subset, split=split)
    return ds


def load_common_voice(subset: str = "ka", split: str = "train",
                       max_hours: float | None = None) -> "Dataset":
    """Load Common Voice Georgian (optional)."""
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        subset,
        split=split,
        token=token,
    )
    # Rename columns to match FLEURS convention
    if "sentence" in ds.column_names:
        ds = ds.rename_column("sentence", "transcription")
    if max_hours is not None:
        # Estimate total duration and subsample
        max_samples = int(max_hours * 3600 / 5)  # assume ~5 s avg
        if len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
    return ds


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _filter_by_duration(example: dict, max_duration_s: float) -> bool:
    """Return True if clip duration <= max_duration_s."""
    audio = example["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    return duration <= max_duration_s


def _check_and_remove_overlap(train_ds, test_ds, text_col: str = "transcription"):
    """Remove training examples whose transcript appears in the test set."""
    test_texts = set(normalize_text(t) for t in test_ds[text_col])
    n_before = len(train_ds)

    def _not_in_test(example):
        return normalize_text(example[text_col]) not in test_texts

    train_ds = train_ds.filter(_not_in_test)
    n_removed = n_before - len(train_ds)
    if n_removed > 0:
        print(f"⚠️  Removed {n_removed} training examples overlapping with test transcripts.")
    else:
        print("✅ No train/test transcript overlap found.")
    return train_ds


def prepare_dataset(
    processor: WhisperProcessor,
    config: dict,
) -> dict:
    """Full data preparation pipeline.

    Returns a dict with keys: 'train', 'validation', 'test', each a HF Dataset.
    """
    data_cfg = config["data"]
    sr = data_cfg["sampling_rate"]
    max_dur = data_cfg["max_duration_s"]

    # 1. Load FLEURS splits
    print("📥 Loading FLEURS Georgian…")
    train_ds = load_fleurs(data_cfg["subset"], "train")
    val_ds = load_fleurs(data_cfg["subset"], "validation")
    test_ds = load_fleurs(data_cfg["subset"], "test")

    # 2. Optionally load Common Voice and merge into training
    cv_cfg = data_cfg.get("common_voice", {})
    if cv_cfg and cv_cfg.get("enabled", False):
        print("📥 Loading Common Voice Georgian…")
        cv_train = load_common_voice(
            cv_cfg["subset"], "train", cv_cfg.get("max_hours")
        )
        # Keep only columns that exist in both
        common_cols = list(set(train_ds.column_names) & set(cv_train.column_names))
        train_ds = concatenate_datasets([
            train_ds.select_columns(common_cols),
            cv_train.select_columns(common_cols),
        ])
        print(f"   Combined training set: {len(train_ds)} examples")

    # 3. Cast audio to target sampling rate
    print(f"🔊 Resampling audio to {sr} Hz…")
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=sr))
    val_ds = val_ds.cast_column("audio", Audio(sampling_rate=sr))
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=sr))

    # 4. Filter long clips
    print(f"✂️  Dropping clips longer than {max_dur}s…")
    n_before = len(train_ds)
    train_ds = train_ds.filter(partial(_filter_by_duration, max_duration_s=max_dur))
    val_ds = val_ds.filter(partial(_filter_by_duration, max_duration_s=max_dur))
    test_ds = test_ds.filter(partial(_filter_by_duration, max_duration_s=max_dur))
    print(f"   Train: {n_before} → {len(train_ds)}")

    # 5. Remove train/test overlap
    train_ds = _check_and_remove_overlap(train_ds, test_ds)

    # 6. Feature extraction & tokenization
    print("🔧 Extracting features and tokenizing…")

    def _prepare_features(batch):
        audio = batch["audio"]
        # Compute log-Mel spectrogram
        inputs = processor.feature_extractor(
            audio["array"], sampling_rate=sr, return_tensors="np"
        )
        # Tokenize the transcription
        batch["input_features"] = inputs.input_features[0]
        batch["labels"] = processor.tokenizer(
            batch["transcription"],
            truncation=True,
            max_length=448,
        ).input_ids
        return batch

    train_ds = train_ds.map(_prepare_features, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(_prepare_features, remove_columns=val_ds.column_names)
    # Keep test_ds raw for evaluation (we need transcription & audio)

    print(f"✅ Data ready — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return {"train": train_ds, "validation": val_ds, "test": test_ds}


# ---------------------------------------------------------------------------
# Data collator for Whisper
# ---------------------------------------------------------------------------

class WhisperDataCollator:
    """Collate examples into padded batches for Seq2SeqTrainer."""

    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict:
        import torch

        # Pad input features (log-Mel spectrograms)
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels (token IDs) — replace padding with -100 for loss masking
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if the model prepends it during generation
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


if __name__ == "__main__":
    # Quick test
    cfg = load_config()
    processor = WhisperProcessor.from_pretrained(cfg["model"]["name"])
    datasets = prepare_dataset(processor, cfg)
    print("Columns (train):", datasets["train"].column_names)
    print("Columns (test):", datasets["test"].column_names)
