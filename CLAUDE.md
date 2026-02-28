# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VLM Tutorial - Vision Language Model learning project. A hands-on guide to understanding and implementing VLMs.

## Project Structure

```
vlm_tutorial/
├── notebooks/          # Jupyter notebooks for learning and experiments
├── src/               # Source code
│   ├── models/        # VLM model implementations (CLIP, BLIP, LLaVA)
│   ├── datasets/      # Dataset loaders and processors (COCO, VQA)
│   └── utils/         # Utility functions
├── scripts/           # Training and inference scripts (TODO)
└── tests/             # Unit tests (TODO)
```

## Current Status

**Project is in initial setup phase.** Most source files are placeholders (`__init__.py` only).

### Implemented Modules (imports only):
- `src.models`: Trainer, Evaluator, visualize_attention (to be implemented)
- `src.datasets`: COCODataset, VQADataset, prepare_transforms (to be implemented)
- `src.utils`: Utility functions (to be implemented)

### Planned Models:
- CLIPModel (CLIP)
- BLIPModel (BLIP)
- LLaVAModel (LLaVA)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Key Dependencies

- **PyTorch** (>=2.0.0) - Core deep learning framework
- **Transformers** (>=4.30.0) - Hugging Face model library
- **Datasets** (>=2.14.0) - Dataset loading
- **PEFT** (>=0.5.0) - Parameter-Efficient Fine-Tuning
- **bitsandbytes** (>=0.41.0) - Low-bit quantization

## Learning Path

1. **Basics** - Understand VLM architecture (notebooks/01_vlm_basics.ipynb)
2. **Pretrained models** - Use Hugging Face models
3. **Fine-tuning** - Custom dataset training
4. **Advanced** - Video-language models, multi-modal LLMs
