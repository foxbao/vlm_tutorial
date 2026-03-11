#!/usr/bin/env python3
"""
CLIP Network Architecture Diagram
===================================

This script generates a visual architecture diagram for CLIP (Contrastive Language-Image Pre-training)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

# Create figure
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title
ax.text(50, 95, 'CLIP Architecture', ha='center', fontsize=20, fontweight='bold')
ax.text(50, 91, 'Contrastive Language-Image Pre-training', ha='center', fontsize=10, style='italic')

# Color scheme
color_vision = '#4A90E2'
color_text = '#50E3C2'
color_shared = '#F5A623'

# Vision Branch
vision_x = 20
vision_y_start = 70
vision_layers = ['Patch Embedding', 'LayerNorm', 'Vision Transformer\n(12 layers)']

# Text Branch
text_x = 80
text_y_start = 70
text_layers = ['Text Embedding', 'LayerNorm', 'Text Transformer\n(12 layers)']

# Shared components
shared_x = 50
shared_y = 40
shared_components = ['Logits for\n(V, T) pairs', 'Negative\nLog-Likelihood Loss']

# Draw Vision Branch
ax.text(vision_x, vision_y_start, 'VISUAL', fontweight='bold', color=color_vision,
        ha='center', va='center', fontsize=10)
ax.add_patch(FancyBboxPatch((vision_x - 8, vision_y_start - 15), 16, 30,
                            boxstyle="round,pad=0.1", ec=color_vision, fc='none'))

for i, layer in enumerate(reversed(vision_layers)):
    y_pos = vision_y_start - i * 4
    ax.text(vision_x, y_pos, layer, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color_vision, alpha=0.15),
            fontsize=8)
    if i > 0:
        # Arrow pointing up in vision branch
        arrow = FancyArrowPatch((vision_x, y_pos + 4), (vision_x, y_pos + 1),
                               arrowstyle='->', mutation_scale=12, color=color_vision, lw=2)
        ax.add_patch(arrow)
    else:
        # Input arrow from outside
        arrow = FancyArrowPatch((vision_x, y_pos + 4), (vision_x, y_pos + 1),
                               arrowstyle='->', mutation_scale=12, color='#666666', lw=2)
        ax.add_patch(arrow)
        ax.text(vision_x - 12, y_pos + 3, 'Image Input\n(Input: 224x224x3)',
                ha='right', va='center', fontsize=8, color='#666666')

# Draw Text Branch
ax.text(text_x, text_y_start, 'TEXT', fontweight='bold', color=color_text,
        ha='center', va='center', fontsize=10)
ax.add_patch(FancyBboxPatch((text_x - 8, text_y_start - 15), 16, 30,
                            boxstyle="round,pad=0.1", ec=color_text, fc='none'))

for i, layer in enumerate(reversed(text_layers)):
    y_pos = text_y_start - i * 4
    ax.text(text_x, y_pos, layer, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color_text, alpha=0.15),
            fontsize=8)
    if i > 0:
        # Arrow pointing up in text branch
        arrow = FancyArrowPatch((text_x, y_pos + 4), (text_x, y_pos + 1),
                               arrowstyle='->', mutation_scale=12, color=color_text, lw=2)
        ax.add_patch(arrow)
    else:
        # Input arrow from outside
        arrow = FancyArrowPatch((text_x, y_pos + 4), (text_x, y_pos + 1),
                               arrowstyle='->', mutation_scale=12, color='#666666', lw=2)
        ax.add_patch(arrow)
        ax.text(text_x + 12, y_pos + 3, 'Text Token IDs\n(Input: tokenized text)',
                ha='left', va='center', fontsize=8, color='#666666')

# Draw Shared Components (output layer)
ax.text(shared_x, shared_y, 'SHARED OUTPUT', fontweight='bold', color=color_shared,
        ha='center', va='center', fontsize=10)
ax.add_patch(FancyBboxPatch((shared_x - 9, shared_y - 8), 18, 16,
                            boxstyle="round,pad=0.1", ec=color_shared, fc='none'))

# Output labels
y_output = shared_y
ax.text(shared_x, y_output, shared_components[0],
        ha='center', va='center', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=color_shared, alpha=0.15))

# Output arrows to loss function
ax.text(50, y_output - 3, 'Shared projection\nfeatures (512-dim each)',
        ha='center', va='center', fontsize=8, color='#555555')

# Connect vision and text to shared
# Vision -> Shared
arrow_vision = FancyArrowPatch((vision_x, vision_y_start), (shared_x - 4, shared_y + 4),
                               arrowstyle='->', mutation_scale=15, color=color_vision, lw=2, linestyle='--')
ax.add_patch(arrow_vision)
# Text -> Shared
arrow_text = FancyArrowPatch((text_x, text_y_start), (shared_x + 4, shared_y + 4),
                              arrowstyle='->', mutation_scale=15, color=color_text, lw=2, linestyle='--')
ax.add_patch(arrow_text)

# Shared -> Loss
arrow_loss = FancyArrowPatch((shared_x, shared_y - 8), (shared_x, shared_y - 12),
                              arrowstyle='->', mutation_scale=15, color='#666666', lw=2, linestyle=':')
ax.add_patch(arrow_loss)
ax.text(shared_x, shared_y - 15, 'Cross-Entropy Loss\n(NLL Loss)', ha='center', va='top',
        fontsize=8, color='#666666')

# Legend and explanations
legend_x = 50
legend_y = 15

explanation_text = """
Key Components:
• Vision Encoder: ViT (Vision Transformer) processes image patches
  - Input: 224×224 RGB image
  - Output: Image embeddings (512-dim)
• Text Encoder: Transformer processes text tokens
  - Input: Tokenized text sequences
  - Output: Text embeddings (512-dim)
• Shared Projection: Projects both to same feature space
• Contrastive Learning: Matches image-text pairs, discriminates negatives

Architecture Type:
Dual-encoder with contrastive pre-training
"""

ax.text(legend_x, legend_y + 3, explanation_text, ha='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', edgecolor='#cccccc'))

# Component sizes
ax.text(20, 8, 'Model: CLIP-ViT-B/32',
        ha='center', fontsize=9, fontweight='bold', color='#333333')
ax.text(20, 5, 'Parameters: ~151M',
        ha='center', fontsize=8, color='#666666')

ax.text(80, 8, 'Tokenized text → Embedding',
        ha='center', fontsize=9, fontweight='bold', color='#333333')
ax.text(80, 5, 'Context: text length up to 77 tokens',
        ha='center', fontsize=8, color='#666666')

plt.tight_layout()
output_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(output_dir, 'clip_architecture.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'clip_architecture.pdf'), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'clip_architecture.svg'), format='svg', bbox_inches='tight')

print("CLIP architecture diagram saved to docs/clip/:")
print("  - clip_architecture.png")
print("  - clip_architecture.pdf")
print("  - clip_architecture.svg")

plt.close()