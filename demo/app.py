"""
app.py — Gradio demo for Georgian Whisper Fine-Tune.

Three tabs:
  1. Precomputed Samples — Side-by-side baseline vs fine-tuned on test clips
  2. Live Microphone     — Record and transcribe with both models
  3. Results             — Metrics table, training curves, error analysis
"""

import os
import json
import random

import yaml
import torch
import numpy as np
import gradio as gr
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ---------------------------------------------------------------------------
# Config & globals
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model cache
_models = {}
_processor = None


def get_processor():
    global _processor
    if _processor is None:
        _processor = WhisperProcessor.from_pretrained(CONFIG["model"]["name"])
    return _processor


def get_model(label: str):
    """Load and cache a model."""
    global _models
    if label not in _models:
        if label == "baseline":
            path = CONFIG["model"]["name"]
        elif label == "finetuned":
            path = "checkpoints/final_model"
        else:
            path = label

        print(f"Loading model: {label} ({path})")
        model = WhisperForConditionalGeneration.from_pretrained(path)
        model = model.to(DEVICE)
        model.eval()
        model.generation_config.language = CONFIG["model"]["language"]
        model.generation_config.task = CONFIG["model"]["task"]
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None
        _models[label] = model

    return _models[label]


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(audio_input, model_label: str = "baseline") -> str:
    """Transcribe audio with a given model.

    audio_input: can be a file path (str) or a tuple (sample_rate, numpy_array)
    """
    import librosa

    processor = get_processor()
    model = get_model(model_label)

    if isinstance(audio_input, str):
        # File path
        audio, sr = librosa.load(audio_input, sr=16000, mono=True)
    elif isinstance(audio_input, tuple):
        sr, audio = audio_input
        audio = audio.astype(np.float32)
        # Mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        # Normalize
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
    else:
        return "Error: unsupported audio input"

    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_new_tokens=225)

    text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return text


# ---------------------------------------------------------------------------
# Tab 1: Precomputed samples
# ---------------------------------------------------------------------------

def load_precomputed_samples():
    """Load precomputed sample clips from the test set."""
    csv_path = "results/per_clip.csv"
    if not os.path.exists(csv_path):
        return None

    import pandas as pd
    df = pd.read_csv(csv_path)

    # Get baseline and finetuned results
    baseline = df[df["model"] == "baseline"].head(CONFIG["demo"]["num_precomputed_samples"])
    finetuned = df[df["model"] == "finetuned"]

    samples = []
    for _, b_row in baseline.iterrows():
        ft_row = finetuned[finetuned["id"] == b_row["id"]]
        ft_hyp = ft_row["hypothesis"].values[0] if len(ft_row) > 0 else "N/A"
        ft_wer = ft_row["wer"].values[0] if len(ft_row) > 0 else -1

        samples.append({
            "id": b_row["id"],
            "reference": b_row["reference"],
            "baseline_hyp": b_row["hypothesis"],
            "baseline_wer": b_row["wer"],
            "finetuned_hyp": ft_hyp,
            "finetuned_wer": ft_wer,
        })

    return samples


def format_samples_html(samples: list | None) -> str:
    """Format precomputed samples as HTML."""
    if not samples:
        return "<p>No precomputed samples found. Run evaluation first.</p>"

    html = "<div style='font-family: sans-serif;'>"
    for s in samples:
        b_color = "#ff6b6b" if s["baseline_wer"] > 50 else "#ffd93d" if s["baseline_wer"] > 20 else "#6bcb77"
        f_color = "#ff6b6b" if s["finetuned_wer"] > 50 else "#ffd93d" if s["finetuned_wer"] > 20 else "#6bcb77"

        html += f"""
        <div style='border:1px solid #333; border-radius:12px; padding:16px; margin:12px 0; background:#1a1a2e;'>
            <h4 style='color:#e0e0e0; margin-top:0;'>Clip #{s['id']}</h4>
            <p style='color:#a0a0a0;'><b>Reference:</b> {s['reference']}</p>
            <div style='display:flex; gap:16px;'>
                <div style='flex:1; background:#16213e; padding:12px; border-radius:8px;'>
                    <p style='color:#e0e0e0;'><b>Baseline</b>
                       <span style='background:{b_color}; color:#000; padding:2px 8px; border-radius:4px; margin-left:8px;'>
                       WER: {s['baseline_wer']:.0f}%</span></p>
                    <p style='color:#c0c0c0;'>{s['baseline_hyp']}</p>
                </div>
                <div style='flex:1; background:#0f3460; padding:12px; border-radius:8px;'>
                    <p style='color:#e0e0e0;'><b>Fine-tuned</b>
                       <span style='background:{f_color}; color:#000; padding:2px 8px; border-radius:4px; margin-left:8px;'>
                       WER: {s['finetuned_wer']:.0f}%</span></p>
                    <p style='color:#c0c0c0;'>{s['finetuned_hyp']}</p>
                </div>
            </div>
        </div>
        """
    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Tab 2: Live mic transcription
# ---------------------------------------------------------------------------

def transcribe_live(audio):
    """Transcribe mic input with both models."""
    if audio is None:
        return "No audio recorded.", "No audio recorded."

    baseline_text = transcribe(audio, "baseline")
    finetuned_text = transcribe(audio, "finetuned")

    return baseline_text, finetuned_text


# ---------------------------------------------------------------------------
# Tab 3: Results
# ---------------------------------------------------------------------------

def load_results_html() -> str:
    """Load and format results summary."""
    summary_path = "results/summary.json"
    if not os.path.exists(summary_path):
        return "<p>No results found. Run evaluation first.</p>"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    html = "<div style='font-family: sans-serif; color: #e0e0e0;'>"
    html += f"<h3>Test Set: {summary['test_set']}</h3>"

    # Metrics table
    html += """
    <table style='width:100%; border-collapse:collapse; margin:16px 0;'>
        <tr style='background:#16213e;'>
            <th style='padding:12px; text-align:left; border-bottom:2px solid #333;'>Model</th>
            <th style='padding:12px; text-align:right; border-bottom:2px solid #333;'>WER</th>
            <th style='padding:12px; text-align:right; border-bottom:2px solid #333;'>CER</th>
            <th style='padding:12px; text-align:right; border-bottom:2px solid #333;'>RTF</th>
        </tr>
    """
    for name, m in summary["models"].items():
        html += f"""
        <tr style='border-bottom:1px solid #333;'>
            <td style='padding:10px;'>{name}</td>
            <td style='padding:10px; text-align:right;'>{m['wer']}%</td>
            <td style='padding:10px; text-align:right;'>{m['cer']}%</td>
            <td style='padding:10px; text-align:right;'>{m['rtf']}</td>
        </tr>
        """
    html += "</table>"

    # Training curves image
    curves_path = "results/training_curves.png"
    if os.path.exists(curves_path):
        import base64
        with open(curves_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        html += f"<h3>Training Curves</h3>"
        html += f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%; border-radius:8px;'/>"

    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def create_app():
    """Create and return the Gradio app."""

    with gr.Blocks(
        title="Georgian Whisper Fine-Tune Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .tab-content { min-height: 400px; }
        """
    ) as app:

        gr.Markdown(
            """
            # 🇬🇪 Georgian Whisper Fine-Tune Demo

            Side-by-side comparison of **Whisper-small baseline** vs **fine-tuned on Georgian speech**.
            """
        )

        with gr.Tabs():
            # ------ Tab 1: Precomputed Samples ------
            with gr.Tab("📊 Sample Comparisons"):
                gr.Markdown("### Precomputed baseline vs fine-tuned transcriptions from the test set")
                samples = load_precomputed_samples()
                samples_html = format_samples_html(samples)
                gr.HTML(samples_html)

            # ------ Tab 2: Live Microphone ------
            with gr.Tab("🎤 Live Transcription"):
                gr.Markdown("### Record audio and compare transcriptions in real-time")

                with gr.Row():
                    audio_input = gr.Audio(
                        label="Record or upload Georgian speech",
                        type="filepath",
                        sources=["microphone", "upload"],
                    )

                transcribe_btn = gr.Button("🔍 Transcribe", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Baseline (Whisper-small)")
                        baseline_output = gr.Textbox(
                            label="Baseline transcription",
                            lines=3,
                            interactive=False,
                        )
                    with gr.Column():
                        gr.Markdown("#### Fine-tuned (Whisper-small)")
                        finetuned_output = gr.Textbox(
                            label="Fine-tuned transcription",
                            lines=3,
                            interactive=False,
                        )

                transcribe_btn.click(
                    fn=transcribe_live,
                    inputs=[audio_input],
                    outputs=[baseline_output, finetuned_output],
                )

            # ------ Tab 3: Results ------
            with gr.Tab("📈 Results"):
                gr.Markdown("### Evaluation metrics and training curves")
                results_html = load_results_html()
                gr.HTML(results_html)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Georgian Whisper Demo")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    args = parser.parse_args()

    app = create_app()
    app.launch(share=args.share, server_port=args.port)
