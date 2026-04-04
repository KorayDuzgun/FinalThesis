"""
Generate Figure 3.1: End-to-end Research Framework Overview
for Chapter 3 Methodology
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# --- Thesis style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# --- Colors matching the project palette ---
STAGE_COLORS = {
    'data':         '#4e79a7',  # steel blue
    'preprocess':   '#59a14f',  # green
    'features':     '#76b7b2',  # teal
    'model':        '#f28e2b',  # orange
    'conformal':    '#e15759',  # red
    'segment':      '#b07aa1',  # purple
}

EXPERIMENT_COLORS = {
    'exp1': '#1f77b4',
    'exp2': '#ff7f0e',
    'exp3': '#d62728',
}

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# ============================================================
# MAIN PIPELINE BOXES (top row, left to right)
# ============================================================
box_width = 1.8
box_height = 1.2
y_main = 7.0
x_positions = [0.5, 2.8, 5.1, 7.4, 9.7, 12.0]

stages = [
    ("1. Data\nAcquisition &\nExploration",    'data'),
    ("2. Preprocessing\n& Cleaning",           'preprocess'),
    ("3. Feature\nEngineering",                 'features'),
    ("4. Baseline\nPoint Prediction\n(XGBoost)", 'model'),
    ("5. Conformal\nPrediction\n(Uncertainty)",  'conformal'),
    ("6. Segment-Level\nUncertainty\nDecomposition", 'segment'),
]

# Draw main boxes
for i, (label, color_key) in enumerate(stages):
    x = x_positions[i]
    color = STAGE_COLORS[color_key]

    fancy_box = FancyBboxPatch(
        (x, y_main), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='white',
        linewidth=2, alpha=0.9
    )
    ax.add_patch(fancy_box)
    ax.text(x + box_width/2, y_main + box_height/2, label,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', linespacing=1.3)

# Draw arrows between main boxes
for i in range(len(x_positions) - 1):
    x_start = x_positions[i] + box_width
    x_end = x_positions[i+1]
    y_arrow = y_main + box_height / 2
    ax.annotate('', xy=(x_end, y_arrow), xytext=(x_start, y_arrow),
                arrowprops=dict(arrowstyle='->', color='#333333',
                                lw=2, mutation_scale=15))

# ============================================================
# INPUT / OUTPUT ANNOTATIONS
# ============================================================
# Input: Raw GPS Data
ax.text(x_positions[0] + box_width/2, y_main + box_height + 0.45,
        "Raw Bus GPS\nTrajectories (GTFS)",
        ha='center', va='center', fontsize=8.5, fontstyle='italic',
        color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                  edgecolor='#cccccc', linewidth=1))
ax.annotate('', xy=(x_positions[0] + box_width/2, y_main + box_height),
            xytext=(x_positions[0] + box_width/2, y_main + box_height + 0.2),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))

# Output: Uncertainty-Aware Predictions
ax.text(x_positions[5] + box_width/2, y_main + box_height + 0.45,
        "Uncertainty-Aware\nETA Predictions",
        ha='center', va='center', fontsize=8.5, fontstyle='italic',
        color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                  edgecolor='#cccccc', linewidth=1))
ax.annotate('', xy=(x_positions[5] + box_width/2, y_main + box_height + 0.2),
            xytext=(x_positions[5] + box_width/2, y_main + box_height),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))

# ============================================================
# DETAIL BOXES (middle row) - what each stage produces
# ============================================================
y_detail = 5.2
detail_height = 0.9

details = [
    (x_positions[0], "785,976 segment\nrecords, 3 routes,\n55 days"),
    (x_positions[1], "747,798 records\n19,614 trips\n(4.86% removed)"),
    (x_positions[2], "16 route-level\n26 segment-level\nfeatures"),
    (x_positions[3], "Point predictions\n+ residuals\n(nonconformity scores)"),
    (x_positions[4], "Prediction intervals\nwith coverage\nguarantees"),
    (x_positions[5], "Per-segment\nuncertainty\nattribution"),
]

for x, text in details:
    fancy_box = FancyBboxPatch(
        (x, y_detail), box_width, detail_height,
        boxstyle="round,pad=0.08",
        facecolor='#fafafa', edgecolor='#bbbbbb',
        linewidth=1, alpha=0.95
    )
    ax.add_patch(fancy_box)
    ax.text(x + box_width/2, y_detail + detail_height/2, text,
            ha='center', va='center', fontsize=7.5, color='#444444',
            linespacing=1.25)

    # Arrow from main box down to detail
    ax.annotate('', xy=(x + box_width/2, y_detail + detail_height),
                xytext=(x + box_width/2, y_main),
                arrowprops=dict(arrowstyle='->', color='#aaaaaa',
                                lw=1, linestyle='--'))

# ============================================================
# EXPERIMENT MAPPING (bottom section)
# ============================================================
y_exp_title = 3.8
ax.text(7.0, y_exp_title, "Experimental Mapping",
        ha='center', va='center', fontsize=12, fontweight='bold',
        color='#333333')

# Horizontal line separator
ax.plot([1, 13], [3.55, 3.55], color='#dddddd', linewidth=1.5)

# Experiment boxes
y_exp = 2.2
exp_width = 3.5
exp_height = 1.2
exp_gap = 0.35

experiments = [
    {
        'label': 'Experiment 1 (RQ1)',
        'title': 'Static Conformal Prediction\nUnder Distribution Shift',
        'color': EXPERIMENT_COLORS['exp1'],
        'connects_to': 4,  # index in x_positions (stage 5)
    },
    {
        'label': 'Experiment 2 (RQ2)',
        'title': 'Online Adaptive\nConformal Prediction',
        'color': EXPERIMENT_COLORS['exp2'],
        'connects_to': 4,
    },
    {
        'label': 'Experiment 3 (RQ3)',
        'title': 'Segment-Level Uncertainty\nDecomposition & Attribution',
        'color': EXPERIMENT_COLORS['exp3'],
        'connects_to': 5,
    },
]

exp_x_positions = [1.2, 5.2, 9.2]

for i, exp in enumerate(experiments):
    x = exp_x_positions[i]
    color = exp['color']

    # Experiment box
    fancy_box = FancyBboxPatch(
        (x, y_exp), exp_width, exp_height,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor='white',
        linewidth=2, alpha=0.85
    )
    ax.add_patch(fancy_box)

    # Label (top)
    ax.text(x + exp_width/2, y_exp + exp_height - 0.25, exp['label'],
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white')

    # Title (bottom)
    ax.text(x + exp_width/2, y_exp + 0.35, exp['title'],
            ha='center', va='center', fontsize=8, color='white',
            linespacing=1.2)

    # Dashed arrow up to the pipeline stage
    stage_x = x_positions[exp['connects_to']] + box_width / 2
    ax.annotate('',
                xy=(stage_x, y_detail),
                xytext=(x + exp_width/2, y_exp + exp_height),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, linestyle='--',
                                connectionstyle='arc3,rad=0.0'))

# ============================================================
# KEY at bottom
# ============================================================
y_key = 0.8
ax.text(1.2, y_key, "Key:",
        ha='left', va='center', fontsize=9, fontweight='bold', color='#555555')

key_items = [
    ("Pipeline Stage", '#4e79a7', 2.5),
    ("Stage Output", '#fafafa', 5.5),
    ("Experiment", '#ff7f0e', 8.5),
]

for label, color, kx in key_items:
    ec = '#bbbbbb' if color == '#fafafa' else 'white'
    tc = '#444444' if color == '#fafafa' else 'white'
    box = FancyBboxPatch(
        (kx, y_key - 0.2), 1.0, 0.4,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=ec, linewidth=1, alpha=0.9
    )
    ax.add_patch(box)
    ax.text(kx + 1.2, y_key, label,
            ha='left', va='center', fontsize=8, color='#555555')

# ============================================================
# SAVE
# ============================================================
output_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(output_dir, 'fig_3_1_framework_overview.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(output_dir, 'fig_3_1_framework_overview.pdf'),
            bbox_inches='tight', facecolor='white')
print(f"Saved to {output_dir}/fig_3_1_framework_overview.png and .pdf")
plt.close()