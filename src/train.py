"""
train.py — Fine-tune Whisper-small on Georgian speech (FLEURS ka_ge).

Features:
  - Config-driven (config.yaml)
  - fp16 training on GPU
  - Language forced to Georgian
  - WER-based best-checkpoint selection
  - Auto-resume from latest checkpoint (local or Drive)
  - Wall-clock time limit for free Colab sessions
  - Checkpoints synced to Google Drive
"""

import os
import sys
import glob
import time
import shutil

import yaml
import torch
import evaluate
import numpy as np
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)

from src.data_prep import load_config, prepare_dataset, WhisperDataCollator


# ---------------------------------------------------------------------------
# WER compute function for the trainer
# ---------------------------------------------------------------------------

def make_compute_metrics(processor: WhisperProcessor):
    """Return a compute_metrics callable for Seq2SeqTrainer."""
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 (ignored index) with pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ---------------------------------------------------------------------------
# Wall-clock callback
# ---------------------------------------------------------------------------

class WallClockCallback(TrainerCallback):
    """Stop training after a maximum wall-clock time."""

    def __init__(self, max_seconds: int):
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time and (time.time() - self.start_time) > self.max_seconds:
            print(f"\n⏱️  Wall-clock limit ({self.max_seconds}s) reached. Stopping training.")
            control.should_training_stop = True
            control.should_save = True


# ---------------------------------------------------------------------------
# Drive sync callback
# ---------------------------------------------------------------------------

class DriveSyncCallback(TrainerCallback):
    """Copy new checkpoints to Google Drive after each save."""

    def __init__(self, local_dir: str, drive_dir: str):
        self.local_dir = local_dir
        self.drive_dir = drive_dir

    def on_save(self, args, state, control, **kwargs):
        if not os.path.isdir("/content/drive"):
            return  # Drive not mounted
        try:
            os.makedirs(self.drive_dir, exist_ok=True)
            # Find the latest checkpoint
            ckpts = sorted(
                glob.glob(os.path.join(self.local_dir, "checkpoint-*")),
                key=os.path.getmtime,
            )
            if ckpts:
                latest = ckpts[-1]
                dest = os.path.join(self.drive_dir, os.path.basename(latest))
                if not os.path.exists(dest):
                    print(f"  📤 Syncing {os.path.basename(latest)} to Drive…")
                    shutil.copytree(latest, dest)
        except Exception as e:
            print(f"  ⚠️  Drive sync failed: {e}")


# ---------------------------------------------------------------------------
# Find the latest checkpoint for resume
# ---------------------------------------------------------------------------

def find_latest_checkpoint(local_dir: str, drive_dir: str) -> str | None:
    """Find the latest checkpoint in local or Drive directories."""
    candidates = []

    for d in [local_dir, drive_dir]:
        if os.path.isdir(d):
            ckpts = glob.glob(os.path.join(d, "checkpoint-*"))
            candidates.extend(ckpts)

    if not candidates:
        return None

    # Sort by checkpoint number
    def _ckpt_num(path):
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return 0

    latest = max(candidates, key=_ckpt_num)

    # If the latest is on Drive, copy it locally
    if latest.startswith(drive_dir) and not latest.startswith(local_dir):
        local_copy = os.path.join(local_dir, os.path.basename(latest))
        if not os.path.exists(local_copy):
            print(f"  📥 Restoring checkpoint from Drive: {os.path.basename(latest)}")
            shutil.copytree(latest, local_copy)
        latest = local_copy

    return latest


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config_path: str = "config.yaml"):
    """Run the full fine-tuning pipeline."""
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]

    print("🏋️ Georgian Whisper Fine-Tuning")
    print("=" * 60)

    # 1. Load model & processor
    print(f"📦 Loading model: {model_cfg['name']}")
    processor = WhisperProcessor.from_pretrained(model_cfg["name"])
    model = WhisperForConditionalGeneration.from_pretrained(model_cfg["name"])

    # Force Georgian language and transcription task
    model.generation_config.language = model_cfg["language"]
    model.generation_config.task = model_cfg["task"]
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Parameters: {n_params:.0f}M")

    # 2. Prepare data
    datasets = prepare_dataset(processor, config)

    # 3. Output directories
    local_dir = train_cfg["output_dir"]
    drive_dir = train_cfg["drive_output_dir"]
    os.makedirs(local_dir, exist_ok=True)

    # 4. Check for existing checkpoint
    resume_from = find_latest_checkpoint(local_dir, drive_dir)
    if resume_from:
        print(f"🔄 Resuming from: {resume_from}")
    else:
        print("🆕 Starting fresh training run.")

    # 5. Training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=local_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        fp16=train_cfg["fp16"] and torch.cuda.is_available(),
        eval_strategy=train_cfg["evaluation_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        logging_steps=train_cfg["logging_steps"],
        predict_with_generate=train_cfg["predict_with_generate"],
        generation_max_length=train_cfg["generation_max_length"],
        report_to=train_cfg["report_to"],
        seed=train_cfg["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        push_to_hub=False,
    )

    # 6. Callbacks
    callbacks = [
        WallClockCallback(train_cfg["max_wall_clock_s"]),
        DriveSyncCallback(local_dir, drive_dir),
    ]

    # 7. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=WhisperDataCollator(processor),
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
    )

    # 8. Train
    print("\n🚀 Starting training…")
    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # 9. Save final model
    final_model_dir = os.path.join(local_dir, "final_model")
    print(f"\n💾 Saving final model to {final_model_dir}")
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)

    # Also save to Drive
    if os.path.isdir("/content/drive"):
        drive_final = os.path.join(drive_dir, "final_model")
        if not os.path.exists(drive_final):
            print(f"  📤 Syncing final model to Drive…")
            shutil.copytree(final_model_dir, drive_final)

    # 10. Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\n✅ Training complete!")
    print(f"   Total steps: {metrics.get('train_steps', 'N/A')}")
    print(f"   Final loss:  {metrics.get('train_loss', 'N/A'):.4f}")

    return trainer, processor


# ---------------------------------------------------------------------------
# Training curves visualization
# ---------------------------------------------------------------------------

def plot_training_curves(log_history: list, output_path: str = "results/training_curves.png"):
    """Plot training loss and validation WER from trainer log history."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract training loss and eval WER
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_wers = []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_wer" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_wers.append(entry["eval_wer"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax1.plot(train_steps, train_losses, color="#2196F3", linewidth=1.5, alpha=0.8)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    # Validation WER
    if eval_wers:
        ax2.plot(eval_steps, eval_wers, color="#FF5722", linewidth=2, marker="o", markersize=4)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("WER (%)")
        ax2.set_title("Validation WER")
        ax2.grid(True, alpha=0.3)
        best_wer = min(eval_wers)
        best_step = eval_steps[eval_wers.index(best_wer)]
        ax2.annotate(
            f"Best: {best_wer:.1f}%\n(step {best_step})",
            xy=(best_step, best_wer),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training curves saved to {output_path}")


if __name__ == "__main__":
    trainer, processor = train()
    plot_training_curves(trainer.state.log_history)
