# 🎥 Video Script — Georgian Whisper Fine-Tune Demo

**Target length:** 60–90 seconds  
**Format:** Screen recording with voiceover (or text captions)

---

## Scene 1: Introduction (10s)

**Show:** Project README on GitHub

**Say:**
> "I fine-tuned OpenAI's Whisper-small model for Georgian speech recognition
> using the FLEURS dataset. Here's a quick demo of the results."

---

## Scene 2: The Problem (10s)

**Show:** Baseline transcription failing on a Georgian clip (Demo Tab 1)

**Say:**
> "Out of the box, Whisper-small struggles with Georgian.
> Here's the baseline trying to transcribe a Georgian sentence —
> it gets a word error rate of [X]%."

---

## Scene 3: The Solution (10s)

**Show:** Fine-tuned transcription on the same clip

**Say:**
> "After fine-tuning on just [N] hours of Georgian speech data,
> the same model drops to [Y]% WER — a [Z]% relative improvement."

---

## Scene 4: Results Table (10s)

**Show:** Demo Tab 3 — Results tab with metrics table

**Say:**
> "On the full test set: baseline WER [X]%, fine-tuned WER [Y]%.
> Character error rate dropped from [A]% to [B]%.
> Both models run faster than real-time on a T4 GPU."

---

## Scene 5: Live Demo (20s)

**Show:** Demo Tab 2 — Record yourself saying something in Georgian

**Say:**
> "Let me try it live. I'll record a sentence in Georgian…
> [Record: say a simple Georgian sentence]
> The baseline says [read baseline output].
> The fine-tuned model says [read fine-tuned output].
> Much closer to what I actually said."

---

## Scene 6: Training Details (10s)

**Show:** Training curves (results/training_curves.png)

**Say:**
> "Training took about [T] hours on a free Colab T4 GPU.
> The validation WER steadily decreased and leveled off around epoch [E]."

---

## Scene 7: Wrap-up (10s)

**Show:** GitHub repository overview

**Say:**
> "The full pipeline is reproducible — just open the Colab notebook,
> add your Hugging Face token, and run all cells.
> Code and instructions are on GitHub. Thanks for watching!"

---

## Recording Tips

1. **Resolution:** 1920×1080 or 1280×720
2. **Audio:** Use a decent microphone, minimize background noise
3. **Pacing:** Don't rush — clear, steady narration
4. **Upload:** YouTube as unlisted, embed link in README
5. **Tool:** OBS Studio (free), or Loom for quick recording
