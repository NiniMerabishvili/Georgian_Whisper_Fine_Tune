Demo Video:

https://github.com/user-attachments/assets/71f9688e-512b-44f2-a748-ea4ee59e37c1



# 🇬🇪 Georgian Whisper Fine-Tune

Fine-tuning a pretrained **Whisper-small** model for Georgian speech recognition, with a reproducible evaluation and a side-by-side demo.

## Overview

Whisper handles Georgian poorly out of the box. This project explores how much a small, low-budget fine-tuning run can improve it. We fine-tuned a pretrained `openai/whisper-small` on the FLEURS Georgian dataset and evaluated it against the untouched baseline on a held-out test set. The whole pipeline runs on a **free Colab T4 GPU** and can **resume after the session disconnects**.

> **This is fine-tuning, not training from scratch.** All models start from OpenAI's published Whisper weights.

## Method

| Stage | Details |
|-------|---------|
| **Data** | FLEURS Georgian (`ka_ge`): train for fine-tuning, validation for monitoring, test only for final evaluation. Audio resampled to 16 kHz mono, clips >30s dropped, train/test transcript overlap checked and removed. |
| **Preflight** | Automated check: GPU, Drive storage, dataset loading, model loading, 30-step smoke-test with checkpoint save/resume. |
| **Fine-tuning** | Hugging Face `Seq2SeqTrainer` with fp16, language forced to Georgian, validation WER tracked, best checkpoint kept, checkpoints saved to Google Drive with auto-resume and wall-clock time limit. |
| **Evaluation** | WER, CER, and RTF on the held-out test set. Identical text normalization for references and hypotheses (Unicode NFC, lowercase, punctuation stripped, whitespace collapsed). |
| **Error analysis** | Worst clips per model, failure pattern breakdown with examples from `results/per_clip.csv`. |
| **Demo** | Gradio app with precomputed sample clips, live microphone transcription, and a results tab. |

Hyperparameters are in [`config.yaml`](config.yaml).

## Results

| Model | WER | CER | RTF |
|-------|-----|-----|-----|
| Whisper small (baseline) | _TBD_ | _TBD_ | _TBD_ |
| Whisper small (fine-tuned) | _TBD_ | _TBD_ | _TBD_ |

> **Test set:** FLEURS Georgian test split. Metrics will be filled in after training completes.

### Error Analysis

_To be completed after evaluation. Will include 3–4 observations on:_
- _What the baseline gets wrong_
- _What fine-tuning fixed_
- _What still fails (numbers, names, foreign words)_

## Project Structure

```
├── config.yaml              # All hyperparameters & settings
├── requirements.txt         # Python dependencies
├── colab_run.py             # One-script Colab runner
├── src/
│   ├── data_prep.py         # Data loading & preprocessing
│   ├── preflight.py         # Environment checks
│   ├── train.py             # Fine-tuning pipeline
│   ├── evaluate.py          # WER/CER/RTF evaluation
│   └── analysis.py          # Error analysis
├── demo/
│   └── app.py               # Gradio demo
├── VIDEO_SCRIPT.md          # Demo video recording guide
└── results/                 # Generated at runtime
    ├── per_clip.csv
    ├── training_curves.png
    └── summary.json
```

## Run It

### Option 1: Google Colab (Recommended)

1. Upload this project folder to Google Drive at `My Drive/Georgian_Whisper_Fine-Tune/`
2. Open a new Colab notebook → **Runtime → Change runtime type → T4 GPU**
3. Add `HF_TOKEN` (read access) to **Colab Secrets**
4. Run these cells:

```python
# Cell 1: Setup
!python /content/drive/MyDrive/Georgian_Whisper_Fine-Tune/colab_run.py --step setup

# Cell 2: Preflight
!python colab_run.py --step preflight

# Cell 3: Train (keep browser tab open)
!python colab_run.py --step train

# Cell 4: Evaluate
!python colab_run.py --step evaluate

# Cell 5: Analyze
!python colab_run.py --step analyze

# Cell 6: Demo
!python colab_run.py --step demo --share
```

Or run everything at once:
```python
!python colab_run.py --step all --share
```

### Option 2: Local

```bash
pip install -r requirements.txt
python -m src.preflight          # Check environment
python -m src.train              # Fine-tune
python -m src.evaluate           # Evaluate
python -m src.analysis           # Error analysis
python demo/app.py --share       # Launch demo
```

### Resuming After Disconnection

Training automatically saves checkpoints to Google Drive. If Colab disconnects:
1. Reconnect and re-run setup
2. Run `!python colab_run.py --step train` — it will auto-detect the latest checkpoint

## Limitations

- **Small training set** — gains are limited and overfitting is possible
- **One test set** of read speech — results may differ on spontaneous speech, noise, or dialects
- **WER depends on normalization** choices (Unicode NFC, lowercase, punctuation stripped, whitespace collapsed)
- **Single training run** — no confidence intervals

## Tech Stack

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — Whisper model & Seq2SeqTrainer
- [FLEURS](https://huggingface.co/datasets/google/fleurs) — Georgian speech dataset
- [Gradio](https://gradio.app/) — Interactive demo
- [jiwer](https://github.com/jitsi/jiwer) — WER/CER computation

## License

This project is for educational and research purposes. The Whisper model weights are subject to [OpenAI's license](https://github.com/openai/whisper/blob/main/LICENSE). FLEURS data is released under CC-BY.
