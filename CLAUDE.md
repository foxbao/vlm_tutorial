# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VLM Tutorial - A hands-on Vision Language Model learning project covering CLIP, BLIP, and LLaVA. The project is in early development: training scripts for CLIP are functional, `src/` modules are mostly placeholders.

## Setup & Commands

```bash
# Environment setup (conda env: clip-tutorial)
conda activate clip-tutorial
pip install -r requirements.txt

# Alternative: venv
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Clean environment run (unsets proxies, activates conda)
bash run_clean.sh python <script.py>

# Training
python scripts/clip/clip_train.py    # CLIP training on CIFAR-10 subset (requires CUDA)

# Inference test
python scripts/clip/clip_test.py     # CLIP inference with cached local models

# Notebooks
jupyter notebook notebooks/

# Lint & format (not yet configured, but conventions target these)
ruff check . && ruff format .
black .

# Tests (not yet implemented)
pytest tests/ -v
```

## Architecture

### Scripts & Utilities
- **`scripts/clip/clip_train.py`** — Full CLIP training pipeline: loads `openai/clip-vit-base-patch32`, trains on CIFAR-10 (800 samples), uses `ContrastiveLoss` with temperature=0.07, mixed precision, saves checkpoints to `outputs/clip/`
- **`scripts/clip/clip_test.py`** — CLIP inference: loads local cached model, computes image-text cosine similarity
- **`docs/clip/`** — CLIP architecture diagrams (png/pdf/svg), `plot_clip_architecture.py`, `CLIP架构图.md`
- **`config.py`** — Argparse-based training config (model, data, training hyperparams, PEFT settings)
- **`model_loader.py`** — `load_blip_model()` and `load_clip_model()` with auto device detection
- **`error_handler.py`** — Validation utilities for training config, model, dataset, device; safe save/load

### `src/` (placeholder modules)
- `src/models/` — CLIPModel, BLIPModel, LLaVAModel (imports defined, implementations TODO)
- `src/datasets/` — COCODataset, VQADataset, prepare_transforms (TODO)
- `src/utils/` — Trainer, Evaluator, visualize_attention (TODO)

### Training Data Flow
```
Dataset → DataLoader → CLIPProcessor → Model (image + text features)
→ L2 normalize → ContrastiveLoss → AdamW → Checkpoint
```

## Code Style (from AGENTS.md)

- **2 spaces** indentation, line length < 100
- **Type hints** required on all function args and returns
- **Imports**: alphabetical, grouped as stdlib → third-party → local
- **Naming**: PascalCase (classes), snake_case (functions/vars), UPPER_SNAKE_CASE (constants)
- **Error handling**: catch `Exception` (never bare `except`), include context in messages
- **Comments**: English for code; Chinese allowed for explanatory clarity
- **File layout**: module docstring → imports → constants → classes → helpers → main → `__main__`

## Project-Specific Patterns

- Use `model.get_image_features()` / `model.get_text_features()` for CLIP
- Always check CUDA: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Use `tqdm` for progress bars
- Support offline mode via `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- Save checkpoints as: `torch.save({'epoch': ..., 'state_dict': ...}, path)`

## Known Issues (from .sisyphus/ planning docs)

The BLIP training implementation has critical issues flagged for fixing:
- Data leakage (train data used for eval), broken BLEU tracking, dataset config mismatch
- See `.sisyphus/plans/blip-training-improvement.md` for the full improvement plan
