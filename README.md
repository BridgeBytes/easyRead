---
title: EasyRead Document Simplification
emoji: 📖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
license: mit
short_description: Convert complex text into Easy Read format with pictograms
---

# EasyRead Document Simplification

Convert complex documents into **Easy Read** format — a method of presenting information in a clear, simple, and easy-to-understand way, especially designed for people who have difficulty reading standard written information.

## How It Works

1. **Paste your text** — articles, reports, letters, anything
2. **AI simplifies it** — Google Gemini breaks it into short, clear sentences
3. **Review & edit** — check the sentences and image prompts before generating icons
4. **Generate icons** — pictograms are sourced from the [Global Symbols](https://globalsymbols.com/) AAC library (with GPU-accelerated local fallback via Stable Diffusion)
5. **Export** — download as a Word `.docx` or Markdown file

## Setup

Set the following **Space secret**:

| Secret | Description |
|---|---|
| `GOOGLE_API_KEY` | Your Google Gemini API key ([get one here](https://aistudio.google.com/)) |

## LoRA Weights (Optional)

The Space falls back to **Stable Diffusion v1.5** for local icon generation when a symbol is not found in the Global Symbols library.
To use the fine-tuned EasyRead LoRA style, place your adapter files in the `weights/` directory:

- `weights/adapter_config.json`
- `weights/adapter_model.safetensors` (or `.bin`)

If no weights are found, the base SD v1.5 model is used automatically.

## Hardware

- **Recommended:** GPU (T4 or better) for fast local icon generation
- **CPU:** Works, but icon generation will be slow — the Global Symbols API is always fast regardless of hardware

## Repository Layout

```
app.py              ← Gradio app (entry point)
requirements.txt    ← Dependencies
config/             ← AI prompt configurations (YAML)
weights/            ← Drop LoRA adapter files here (optional)
```

## Tech Stack

- [Gradio](https://gradio.app/) — UI
- [Google Gemini](https://ai.google.dev/) — text simplification (3-pass: simplify → validate → revise)
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) + LoRA — local icon generation fallback
- [Global Symbols API](https://globalsymbols.com/api) — AAC pictogram library (primary icon source)
