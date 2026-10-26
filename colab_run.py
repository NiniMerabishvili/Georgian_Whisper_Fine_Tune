"""
colab_run.py — One-script Colab runner for Georgian Whisper Fine-Tune.

Usage in Google Colab:
  1. Upload this entire project to Google Drive
  2. Open a Colab notebook with T4 GPU
  3. Add HF_TOKEN to Colab Secrets
  4. Run the cells below (copy-paste or !python colab_run.py)

Each section is marked with # === CELL N === for easy copy-paste into notebook cells.
"""

# === CELL 1: Setup & Install ===
# Run this cell first to install dependencies and mount Drive.

import subprocess
import sys
import os

def setup():
    """Install dependencies and mount Google Drive."""

    # Mount Google Drive
    try:
        import os
        if os.path.ismount("/content/drive"):
            print("✅ Google Drive already mounted")
        else:
            from google.colab import drive
            drive.mount("/content/drive")
            print("✅ Google Drive mounted")
    except Exception as e:
        if os.path.ismount("/content/drive"):
            print("✅ Google Drive already mounted")
        else:
            print(f"⚠️  Could not mount Google Drive: {e}")
            print("   Please run: from google.colab import drive; drive.mount('/content/drive')")

    # Set HF_TOKEN from Colab secrets
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("✅ HF_TOKEN loaded from Colab Secrets")
    except Exception:
        if os.environ.get("HF_TOKEN"):
            print("✅ HF_TOKEN found in environment")
        else:
            print("⚠️  HF_TOKEN not found — some features may not work")

    # Clone or navigate to project
    project_dir = "/content/Georgian_Whisper_Fine-Tune"
    drive_project = "/content/drive/MyDrive/Work/NiniProjs/Georgian_Whisper_Fine-Tune"

    if os.path.isdir(drive_project):
        # Use Drive copy
        if not os.path.isdir(project_dir):
            os.symlink(drive_project, project_dir)
        print(f"✅ Project linked from Drive: {drive_project}")
    elif not os.path.isdir(project_dir):
        print("❌ Project not found. Upload it to Google Drive first.")
        print(f"   Expected at: {drive_project}")
        return False

    os.chdir(project_dir)
    print(f"📂 Working directory: {os.getcwd()}")

    # Install dependencies
    print("📦 Installing dependencies…")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "-r", "requirements.txt"
    ])
    print("✅ Dependencies installed")

    return True


# === CELL 2: Preflight Checks ===

def run_preflight():
    """Run all preflight checks."""
    from src.preflight import run_preflight as _preflight
    success = _preflight()
    if not success:
        print("\n⚠️  Fix the issues above before proceeding to training.")
    return success


# === CELL 3: Full Training ===

def run_training():
    """Run the full fine-tuning pipeline."""
    from src.train import train, plot_training_curves
    trainer, processor = train()
    plot_training_curves(trainer.state.log_history)
    print("\n🎉 Training complete! Proceed to evaluation.")
    return trainer


# === CELL 4: Evaluation ===

def run_evaluation():
    """Evaluate baseline and fine-tuned models."""
    from src.evaluate import run_evaluation as _eval
    summary, df = _eval()
    return summary, df


# === CELL 5: Error Analysis ===

def run_analysis():
    """Run error analysis."""
    from src.analysis import analyze_results
    analysis = analyze_results()
    return analysis


# === CELL 6: Launch Demo ===

def run_demo(share=True):
    """Launch the Gradio demo."""
    from demo.app import create_app
    app = create_app()
    app.launch(share=share, debug=True)


# === MAIN: Run everything in sequence ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Georgian Whisper Fine-Tune — Colab Runner")
    parser.add_argument(
        "--step", choices=["setup", "preflight", "train", "evaluate", "analyze", "demo", "all"],
        default="all", help="Which step to run",
    )
    parser.add_argument("--share", action="store_true", help="Share Gradio demo publicly")
    args = parser.parse_args()

    if args.step in ("setup", "all"):
        if not setup():
            sys.exit(1)

    if args.step in ("preflight", "all"):
        if not run_preflight():
            print("\n💡 Fix issues and re-run: python colab_run.py --step preflight")
            if args.step == "all":
                sys.exit(1)

    if args.step in ("train", "all"):
        run_training()

    if args.step in ("evaluate", "all"):
        run_evaluation()

    if args.step in ("analyze", "all"):
        run_analysis()

    if args.step in ("demo", "all"):
        run_demo(share=args.share)
