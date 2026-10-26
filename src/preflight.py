"""
preflight.py — Automated pre-training checks for the Georgian Whisper Fine-Tune project.

Runs in sequence:
  1. GPU availability & VRAM
  2. Google Drive mount & free space
  3. HF_TOKEN validity (optional)
  4. Dataset loading (FLEURS ka_ge, a few examples)
  5. Model & processor loading
  6. Smoke-test training (N steps, save checkpoint, reload)

Exit code 0 = all checks passed; non-zero = see error messages.
"""

import os
import sys
import time
import shutil
import tempfile

import yaml
import torch
import numpy as np


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _header(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")


def _warn(msg: str):
    print(f"  ⚠️  {msg}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_gpu() -> bool:
    """Check for CUDA GPU and report VRAM."""
    _header("1/6  GPU Check")
    if not torch.cuda.is_available():
        _fail("No CUDA GPU detected. Training will be extremely slow on CPU.")
        return False

    device = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    _ok(f"GPU: {device}  |  VRAM: {vram_gb:.1f} GB")

    if vram_gb < 14:
        _warn("Less than 14 GB VRAM. You may need to reduce batch size.")

    return True


def check_drive(drive_path: str) -> bool:
    """Check Google Drive mount and free space."""
    _header("2/6  Google Drive Check")

    # In Colab, Drive is at /content/drive
    if not os.path.isdir("/content/drive"):
        _warn("Google Drive not mounted. Checkpoints will be saved locally only.")
        _warn("In Colab: run drive.mount('/content/drive') first.")
        return True  # non-fatal

    # Check / create output dir
    os.makedirs(drive_path, exist_ok=True)
    _ok(f"Drive output dir exists: {drive_path}")

    # Free space
    total, used, free = shutil.disk_usage(drive_path)
    free_gb = free / (1024**3)
    _ok(f"Free space: {free_gb:.1f} GB")

    if free_gb < 5:
        _warn(f"Only {free_gb:.1f} GB free. Recommend ≥8 GB for checkpoints.")

    return True


def check_hf_token() -> bool:
    """Check if HF_TOKEN is set and valid."""
    _header("3/6  Hugging Face Token")
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Try Colab secrets
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token
        except Exception:
            pass

    if not token:
        _warn("HF_TOKEN not found. Some datasets (Common Voice) may require it.")
        return True  # non-fatal

    from huggingface_hub import HfApi
    try:
        api = HfApi(token=token)
        user = api.whoami()
        _ok(f"Authenticated as: {user.get('name', user.get('fullname', 'unknown'))}")
    except Exception as e:
        _fail(f"Token validation failed: {e}")
        return False

    return True


def check_dataset(subset: str = "ka_ge") -> bool:
    """Load a few examples from FLEURS to verify dataset access."""
    _header("4/6  Dataset Loading")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "google/fleurs", subset, split="train",
            streaming=True,
        )
        # Grab 3 examples
        examples = list(ds.take(3))
        _ok(f"FLEURS {subset} loaded. Sample transcription:")
        print(f"       \"{examples[0]['transcription'][:80]}…\"")
        return True
    except Exception as e:
        _fail(f"Dataset loading failed: {e}")
        return False


def check_model(model_name: str) -> bool:
    """Load the Whisper model and processor."""
    _header("5/6  Model Loading")
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        _ok(f"Loaded {model_name} ({n_params:.0f}M parameters)")
        del model  # free memory
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        _fail(f"Model loading failed: {e}")
        return False


def check_smoke_test(config: dict) -> bool:
    """Run a short training loop, save a checkpoint, and reload it."""
    _header("6/6  Smoke-Test Training")
    try:
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
        )
        from src.data_prep import prepare_dataset, WhisperDataCollator

        model_name = config["model"]["name"]
        smoke_steps = config["training"]["smoke_test_steps"]

        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

        # Force Georgian
        model.generation_config.language = config["model"]["language"]
        model.generation_config.task = config["model"]["task"]
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None

        # Prepare data
        print("  📥 Loading data for smoke test…")
        datasets = prepare_dataset(processor, config)
        train_ds = datasets["train"]

        # Use a small subset
        subset_size = min(64, len(train_ds))
        train_subset = train_ds.select(range(subset_size))
        val_subset = train_ds.select(range(min(16, len(train_ds))))

        # Temporary output dir
        smoke_dir = tempfile.mkdtemp(prefix="whisper_smoke_")

        args = Seq2SeqTrainingArguments(
            output_dir=smoke_dir,
            max_steps=smoke_steps,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=smoke_steps,
            eval_steps=smoke_steps,
            eval_strategy="steps",
            predict_with_generate=True,
            generation_max_length=50,
            report_to="none",
            seed=42,
            remove_unused_columns=False,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            data_collator=WhisperDataCollator(processor),
            processing_class=processor.feature_extractor,
        )

        print(f"  🏋️ Training for {smoke_steps} steps…")
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        _ok(f"Smoke training done in {elapsed:.1f}s")

        # Save and reload checkpoint
        print("  💾 Saving checkpoint…")
        trainer.save_model(os.path.join(smoke_dir, "smoke_ckpt"))
        reloaded = WhisperForConditionalGeneration.from_pretrained(
            os.path.join(smoke_dir, "smoke_ckpt")
        )
        _ok("Checkpoint saved and reloaded successfully.")

        # Cleanup
        del model, reloaded, trainer
        torch.cuda.empty_cache()
        shutil.rmtree(smoke_dir, ignore_errors=True)

        return True

    except Exception as e:
        _fail(f"Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_preflight(config_path: str = "config.yaml") -> bool:
    """Run all preflight checks. Returns True if all critical checks pass."""
    config = load_config(config_path)
    drive_path = config["training"]["drive_output_dir"]
    model_name = config["model"]["name"]
    subset = config["data"]["subset"]

    print("\n🔍 Running Preflight Checks…")
    print("=" * 60)

    results = {}
    results["gpu"] = check_gpu()
    results["drive"] = check_drive(drive_path)
    results["hf_token"] = check_hf_token()
    results["dataset"] = check_dataset(subset)
    results["model"] = check_model(model_name)

    # Only run smoke test if previous checks passed
    if all(results.values()):
        results["smoke_test"] = check_smoke_test(config)
    else:
        _header("6/6  Smoke-Test Training")
        _warn("Skipped — fix earlier failures first.")
        results["smoke_test"] = False

    # Summary
    print("\n" + "=" * 60)
    print("  PREFLIGHT SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s} {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🎉 All preflight checks passed! Ready to train.")
    else:
        print("\n⚠️  Some checks failed. Fix the issues above before training.")

    return all_pass


if __name__ == "__main__":
    success = run_preflight()
    sys.exit(0 if success else 1)
