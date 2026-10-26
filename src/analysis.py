"""
analysis.py — Error analysis for Georgian Whisper Fine-Tune.

Reads results/per_clip.csv and produces:
  - Top-N worst clips per model
  - Failure pattern breakdown (numbers, names, foreign words, etc.)
  - Summary observations for the README
  - Optionally re-runs evaluation if per_clip.csv is missing
"""

import os
import sys
import re
import json
from collections import Counter

import pandas as pd
import numpy as np

from src.data_prep import load_config


# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------

# Georgian Unicode range: U+10A0–U+10FF (Mkhedruli + Asomtavruli)
_GEORGIAN_RE = re.compile(r"[\u10A0-\u10FF\u2D00-\u2D2F]+")
_DIGIT_RE = re.compile(r"\d+")
_LATIN_RE = re.compile(r"[a-zA-Z]+")


def classify_errors(ref: str, hyp: str) -> list[str]:
    """Classify the types of errors between reference and hypothesis."""
    categories = []

    if not ref.strip():
        return ["empty_reference"]
    if not hyp.strip():
        return ["empty_hypothesis"]

    # Check for digit-related errors
    ref_digits = set(_DIGIT_RE.findall(ref))
    hyp_digits = set(_DIGIT_RE.findall(hyp))
    if ref_digits != hyp_digits and (ref_digits or hyp_digits):
        categories.append("number_error")

    # Check for foreign/Latin word errors
    ref_latin = set(_LATIN_RE.findall(ref))
    hyp_latin = set(_LATIN_RE.findall(hyp))
    if ref_latin or hyp_latin:
        if ref_latin != hyp_latin:
            categories.append("foreign_word_error")

    # Check for major Georgian word mismatches
    ref_words = set(ref.split())
    hyp_words = set(hyp.split())
    missing = ref_words - hyp_words
    extra = hyp_words - ref_words

    if len(missing) > len(ref_words) * 0.5:
        categories.append("major_deletion")
    elif missing:
        categories.append("word_deletion")

    if len(extra) > len(hyp_words) * 0.5:
        categories.append("major_insertion")
    elif extra:
        categories.append("word_insertion")

    # Check for hallucination (hypothesis much longer than reference)
    if len(hyp.split()) > len(ref.split()) * 2 and len(ref.split()) > 2:
        categories.append("hallucination")

    # Check for language confusion (hypothesis has no Georgian chars but ref does)
    ref_has_geo = bool(_GEORGIAN_RE.search(ref))
    hyp_has_geo = bool(_GEORGIAN_RE.search(hyp))
    if ref_has_geo and not hyp_has_geo and hyp.strip():
        categories.append("language_confusion")

    if not categories:
        if ref == hyp:
            categories.append("correct")
        else:
            categories.append("substitution")

    return categories


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(
    csv_path: str = "results/per_clip.csv",
    output_dir: str = "results",
    top_n: int = 10,
) -> dict:
    """Run error analysis on per-clip results."""

    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} not found. Run evaluation first:")
        print("   python -m src.evaluate")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    analysis = {}

    for model_label in df["model"].unique():
        model_df = df[df["model"] == model_label].copy()

        print(f"\n{'='*65}")
        print(f"  Error Analysis: {model_label}")
        print(f"{'='*65}")

        # --- Top-N worst clips ---
        worst = model_df.nlargest(top_n, "wer")
        print(f"\n  📉 Top {top_n} worst clips (by WER):")
        print(f"  {'ID':>6}  {'WER':>8}  {'CER':>8}  Reference (first 50 chars)")
        print(f"  {'-'*60}")
        for _, row in worst.iterrows():
            ref_trunc = row["reference"][:50] + ("…" if len(str(row["reference"])) > 50 else "")
            print(f"  {row['id']:>6}  {row['wer']:>7.1f}%  {row['cer']:>7.1f}%  {ref_trunc}")

        # --- Error categorization ---
        all_categories = Counter()
        model_df["error_categories"] = model_df.apply(
            lambda r: classify_errors(str(r["ref_norm"]), str(r["hyp_norm"])),
            axis=1,
        )
        for cats in model_df["error_categories"]:
            all_categories.update(cats)

        print(f"\n  📊 Error pattern breakdown:")
        total_clips = len(model_df)
        for cat, count in all_categories.most_common():
            pct = count / total_clips * 100
            print(f"     {cat:<25s} {count:>5} ({pct:>5.1f}%)")

        # --- Per-clip examples for worst failures ---
        print(f"\n  📝 Example failures:")
        for idx, (_, row) in enumerate(worst.head(5).iterrows()):
            print(f"\n  [{idx+1}] WER={row['wer']:.0f}%")
            print(f"      REF: {row['reference']}")
            print(f"      HYP: {row['hypothesis']}")
            cats = classify_errors(str(row["ref_norm"]), str(row["hyp_norm"]))
            print(f"      Categories: {', '.join(cats)}")

        # Store for JSON output
        analysis[model_label] = {
            "worst_clips": worst[["id", "wer", "cer", "reference", "hypothesis"]].to_dict("records"),
            "error_breakdown": dict(all_categories.most_common()),
            "total_clips": total_clips,
            "mean_wer": round(model_df["wer"].mean(), 2),
            "median_wer": round(model_df["wer"].median(), 2),
            "std_wer": round(model_df["wer"].std(), 2),
            "correct_count": int(all_categories.get("correct", 0)),
            "correct_pct": round(all_categories.get("correct", 0) / total_clips * 100, 1),
        }

    # --- Comparative analysis (if both models present) ---
    if "baseline" in analysis and "finetuned" in analysis:
        print(f"\n{'='*65}")
        print("  Comparative Analysis: Baseline vs Fine-Tuned")
        print(f"{'='*65}")

        baseline_df = df[df["model"] == "baseline"].set_index("id")
        ft_df = df[df["model"] == "finetuned"].set_index("id")

        common_ids = baseline_df.index.intersection(ft_df.index)

        # Clips that improved
        improved = []
        degraded = []
        unchanged = []

        for clip_id in common_ids:
            b_wer = baseline_df.loc[clip_id, "wer"]
            f_wer = ft_df.loc[clip_id, "wer"]

            if f_wer < b_wer - 1:
                improved.append((clip_id, b_wer, f_wer))
            elif f_wer > b_wer + 1:
                degraded.append((clip_id, b_wer, f_wer))
            else:
                unchanged.append(clip_id)

        print(f"\n  Improved:  {len(improved):>5} clips")
        print(f"  Degraded:  {len(degraded):>5} clips")
        print(f"  Unchanged: {len(unchanged):>5} clips")

        # Show biggest improvements
        if improved:
            improved.sort(key=lambda x: x[1] - x[2], reverse=True)
            print(f"\n  🎯 Biggest improvements:")
            for clip_id, b_wer, f_wer in improved[:5]:
                print(f"     Clip {clip_id}: {b_wer:.0f}% → {f_wer:.0f}%")
                ref = baseline_df.loc[clip_id, "reference"]
                print(f"       REF: {ref[:70]}…" if len(str(ref)) > 70 else f"       REF: {ref}")

        # Show regressions
        if degraded:
            degraded.sort(key=lambda x: x[2] - x[1], reverse=True)
            print(f"\n  ⚠️  Biggest regressions:")
            for clip_id, b_wer, f_wer in degraded[:5]:
                print(f"     Clip {clip_id}: {b_wer:.0f}% → {f_wer:.0f}%")

        analysis["comparison"] = {
            "improved_count": len(improved),
            "degraded_count": len(degraded),
            "unchanged_count": len(unchanged),
            "biggest_improvements": [
                {"id": c[0], "baseline_wer": c[1], "finetuned_wer": c[2]}
                for c in improved[:10]
            ],
            "biggest_regressions": [
                {"id": c[0], "baseline_wer": c[1], "finetuned_wer": c[2]}
                for c in degraded[:10]
            ],
        }

    # Save analysis JSON
    analysis_path = os.path.join(output_dir, "analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n📄 Analysis saved to {analysis_path}")

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Error analysis for Georgian Whisper")
    parser.add_argument("--csv", default="results/per_clip.csv", help="Path to per_clip.csv")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--top-n", type=int, default=10, help="Number of worst clips to show")
    args = parser.parse_args()

    analyze_results(args.csv, args.output, args.top_n)
