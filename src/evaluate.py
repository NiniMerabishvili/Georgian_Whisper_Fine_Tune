"""
evaluate.py — Evaluate baseline and fine-tuned Whisper models on the FLEURS Georgian test set.

Computes:
  - WER  (Word Error Rate)
  - CER  (Character Error Rate)
  - RTF  (Real-Time Factor = processing_time / audio_duration)

Produces:
  - results/per_clip.csv    — per-clip metrics + transcriptions
  - results/summary.json   — aggregate metrics table
"""

import os
import sys
import json
import time

import yaml
import torch
import numpy as np
import pandas as pd
import evaluate as hf_evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

from src.data_prep import load_config, normalize_text


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name_or_path: str,
    processor: WhisperProcessor,
    test_dataset,
    norm_cfg: dict,
    language: str = "ka",
    task: str = "transcribe",
    label: str = "model",
    device: str = "cuda",
) -> pd.DataFrame:
    """Evaluate a single Whisper model on the test set.

    Returns a DataFrame with columns:
      id, reference, hypothesis, ref_norm, hyp_norm, duration_s, inference_s, wer, cer
    """
    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    # Load model
    print(f"\n📦 Loading model: {label} ({model_name_or_path})")
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    model = model.to(device)
    model.eval()

    # Force language
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    rows = []
    print(f"🔍 Evaluating {len(test_dataset)} clips…")

    for i, example in enumerate(tqdm(test_dataset, desc=label)):
        audio = example["audio"]
        reference = example["transcription"]
        duration_s = len(audio["array"]) / audio["sampling_rate"]

        # Prepare input
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features.to(device)

        # Generate
        t0 = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_new_tokens=225)
        inference_s = time.time() - t0

        hypothesis = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

        # Normalize
        ref_norm = normalize_text(reference, norm_cfg)
        hyp_norm = normalize_text(hypothesis, norm_cfg)

        # Per-clip WER and CER
        clip_wer = wer_metric.compute(predictions=[hyp_norm], references=[ref_norm]) * 100 if ref_norm else 0.0
        clip_cer = cer_metric.compute(predictions=[hyp_norm], references=[ref_norm]) * 100 if ref_norm else 0.0

        rows.append({
            "id": example.get("id", i),
            "reference": reference,
            "hypothesis": hypothesis,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "duration_s": duration_s,
            "inference_s": inference_s,
            "wer": clip_wer,
            "cer": clip_cer,
            "model": label,
        })

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_aggregate_metrics(df: pd.DataFrame) -> dict:
    """Compute corpus-level WER, CER, and RTF from per-clip results."""
    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    refs = df["ref_norm"].tolist()
    hyps = df["hyp_norm"].tolist()

    # Filter out empty references
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    refs_clean, hyps_clean = zip(*valid) if valid else ([], [])

    corpus_wer = wer_metric.compute(predictions=list(hyps_clean), references=list(refs_clean)) * 100
    corpus_cer = cer_metric.compute(predictions=list(hyps_clean), references=list(refs_clean)) * 100

    total_audio = df["duration_s"].sum()
    total_inference = df["inference_s"].sum()
    rtf = total_inference / total_audio if total_audio > 0 else 0.0

    return {
        "wer": round(corpus_wer, 2),
        "cer": round(corpus_cer, 2),
        "rtf": round(rtf, 3),
        "num_clips": len(df),
        "total_audio_s": round(total_audio, 1),
        "total_inference_s": round(total_inference, 1),
    }


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    config_path: str = "config.yaml",
    finetuned_model_path: str = "checkpoints/final_model",
    output_dir: str = "results",
):
    """Evaluate baseline and fine-tuned models, save results."""
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    norm_cfg = config["evaluation"]["normalization"]

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test set
    print("📥 Loading FLEURS Georgian test set…")
    test_ds = load_dataset(
        data_cfg["dataset"], data_cfg["subset"],
        split="test",
    )
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=data_cfg["sampling_rate"]))

    # Filter long clips
    def _filter(example):
        return len(example["audio"]["array"]) / example["audio"]["sampling_rate"] <= data_cfg["max_duration_s"]
    test_ds = test_ds.filter(_filter)
    print(f"   Test set: {len(test_ds)} clips")

    # Load processor
    processor = WhisperProcessor.from_pretrained(model_cfg["name"])

    # ---- Baseline evaluation ----
    baseline_df = evaluate_model(
        model_name_or_path=model_cfg["name"],
        processor=processor,
        test_dataset=test_ds,
        norm_cfg=norm_cfg,
        language=model_cfg["language"],
        task=model_cfg["task"],
        label="baseline",
        device=device,
    )

    # ---- Fine-tuned evaluation ----
    finetuned_df = evaluate_model(
        model_name_or_path=finetuned_model_path,
        processor=processor,
        test_dataset=test_ds,
        norm_cfg=norm_cfg,
        language=model_cfg["language"],
        task=model_cfg["task"],
        label="finetuned",
        device=device,
    )

    # ---- Combine and save ----
    all_df = pd.concat([baseline_df, finetuned_df], ignore_index=True)
    csv_path = os.path.join(output_dir, "per_clip.csv")
    all_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n📄 Per-clip results saved to {csv_path}")

    # Aggregate
    baseline_metrics = compute_aggregate_metrics(baseline_df)
    finetuned_metrics = compute_aggregate_metrics(finetuned_df)

    summary = {
        "test_set": f"FLEURS Georgian ({data_cfg['subset']}), {len(test_ds)} clips",
        "models": {
            "Whisper small (baseline)": baseline_metrics,
            "Whisper small (fine-tuned)": finetuned_metrics,
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📊 Summary saved to {summary_path}")

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"  {'Model':<35} {'WER':>8} {'CER':>8} {'RTF':>8}")
    print("-" * 65)
    for name, m in summary["models"].items():
        print(f"  {name:<35} {m['wer']:>7.1f}% {m['cer']:>7.1f}% {m['rtf']:>7.3f}")
    print("=" * 65)

    # WER improvement
    wer_base = baseline_metrics["wer"]
    wer_ft = finetuned_metrics["wer"]
    if wer_base > 0:
        rel_improvement = (wer_base - wer_ft) / wer_base * 100
        print(f"\n  📈 Relative WER improvement: {rel_improvement:.1f}%")
        print(f"     ({wer_base:.1f}% → {wer_ft:.1f}%)")

    return summary, all_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Whisper models on Georgian test set")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", default="checkpoints/final_model", help="Path to fine-tuned model")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    run_evaluation(args.config, args.model, args.output)
