# AGENTS.md - VLM Tutorial

## Build & Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run single test (when tests are added)
pytest tests/ -v

# Lint (when linting setup is added)
ruff check .
flake8 .

# Format code
black .
ruff format .
```

## Code Style Guidelines

### Language & Comments
- Primary language: English for code/APIs
- Comments in English, except where explanatory clarity requires Chinese
- Use inline comments to explain "why" and "how" of complex logic
- Class/docstring comments: Use `"""Docstring"""` format

### Imports
```python
import os
import torch

from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
```
- Alphabetical order with blank line between stdlib and third-party imports
- Group imports: stdlib → third-party → local

### Type Hints
```python
def train_step(
    model: nn.Module,
    processor: CLIPProcessor,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Args:
        model: CLIP model
        processor: CLIP processor
        dataloader: Data loader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epoch: Current epoch
    Returns:
        Average loss
    """
    ...
```
- Always annotate function arguments and return types
- Use `typing` module for complex types (Union, Optional, List, Dict)
- Exported functions must have complete type signatures

### Naming Conventions
- **Classes**: `PascalCase` (`CLIPModel`, `Config`, `ContrastiveLoss`)
- **Functions/Variables**: `snake_case` (`train_step`, `batch_size`, `num_workers`)
- **Constants**: `UPPER_SNAKE_CASE` (`BATCH_SIZE`, `MAX_LEN`)
- **Private members (internal use only)**: `_leading_underscore`

### Function Definition Style
- Group related functions by logical areas (markers: `# ==========...`)
- Put public API first, then internal utilities
- Include clear module docstring at top

### Error Handling
```python
try:
    # Operation that might fail
    ...
except Exception as e:
    print(f"Error: {e}")
    return default_value  # Or `raise` if unrecoverable
```
- Catch `Exception`, not bare `except`
- Include error context in exception message
- For non-critical paths, return `None` or default value with logging
- Never use empty `except: pass`

### Formatting Rules
- 2 spaces indentation
- Line length < 100 characters
- Use `self` for class instance methods
- Keep docstrings concise but complete

### Code Organization
```python
# 1. Module docstring
# 2. Imports
# 3. Constants
# 4. Custom classes
# 5. Helper functions
# 6. Main functions
# 7. If __name__ == "__main__"
```

### Project-Specific Patterns
- CLIPModel: Use `model.get_image_features()` and `model.get_text_features()`
- Always use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Use `tqdm` for progress bars in loops
- Check `os.environ` for offline mode (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)
- Save checkpoints with: `torch.save({'epoch': ..., 'state_dict': ...}, path)`