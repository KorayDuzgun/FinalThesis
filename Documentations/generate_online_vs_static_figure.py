"""
Figure 3.6: Static vs Online Conformal Prediction Comparison
Shows how calibration set management differs between static and online approaches.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1, 1]})

COLORS = {
    'cal': '#59a14f',
    'cal_new': '#76b7b2',
    'test': '#f28e2b',
    'predict': '#4e79a7',
    'stale': '#bab0ac',
    'removed': '#e15759',
    'window': '#b07aa1',
}

days = list(range(1, 26))
day_labels = [f'D{d}' for d in days]

def draw_timeline(ax, title, cal_blocks, test_blocks, annotations=None,
                  removed_blocks=None, window_bracket=None):
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 3)
    ax.axis('off')

    ax.text(0.2, 2.6, title, fontsize=12, fontweight='bold', color='#333333',
            va='center')

    bar_y = 0.8
    bar_h = 1.0
    bw = 0.9

    # Draw blocks
    for d in days:
        x = d * 1.0 + 0.5
        if d in removed_blocks if removed_blocks else []:
            color = COLORS['removed']
            alpha = 0.4
        elif d in cal_blocks:
            color = COLORS['cal']
            alpha = 0.85
        elif d in test_blocks:
            color = COLORS['test']
            alpha = 0.85
        else:
            color = COLORS['stale']
            alpha = 0.3

        rect = FancyBboxPatch(
            (x, bar_y), bw, bar_h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor='white', linewidth=1, alpha=alpha
        )
        ax.add_patch(rect)

        if d <= 7 or d >= 8:
            label = f'D{d}'
            ax.text(x + bw / 2, bar_y - 0.15, label,
                    ha='center', va='top', fontsize=5.5, color='#888888')

    # Window bracket
    if window_bracket:
        wx_start = window_bracket[0] * 1.0 + 0.5
        wx_end = (window_bracket[1] + 1) * 1.0 + 0.5
        bracket_y = bar_y + bar_h + 0.1
        ax.plot([wx_start, wx_start], [bracket_y, bracket_y + 0.2],
                color=COLORS['window'], linewidth=1.5)
        ax.plot([wx_end, wx_end], [bracket_y, bracket_y + 0.2],
                color=COLORS['window'], linewidth=1.5)
        ax.plot([wx_start, wx_end], [bracket_y + 0.2, bracket_y + 0.2],
                color=COLORS['window'], linewidth=1.5)
        mid = (wx_start + wx_end) / 2
        ax.text(mid, bracket_y + 0.35, 'Calibration Window',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=COLORS['window'])

    # Annotations
    if annotations:
        for ann in annotations:
            ax.text(ann['x'], ann['y'], ann['text'],
                    ha=ann.get('ha', 'center'), va='center',
                    fontsize=ann.get('fs', 8), color=ann.get('color', '#666666'),
                    fontstyle=ann.get('style', 'normal'),
                    fontweight=ann.get('weight', 'normal'))

# ============================================================
# Panel A: Static CP — fixed calibration, never updates
# ============================================================
static_cal = list(range(1, 8))  # W4 = days 1-7
static_test = list(range(8, 26))  # W5-W8 = days 8-25

draw_timeline(
    axes[0],
    '(a) Static Conformal Prediction — Fixed Calibration',
    cal_blocks=static_cal,
    test_blocks=static_test,
    annotations=[
        {'x': 4.5, 'y': 0.35, 'text': 'W4 (Calibration)', 'fs': 9,
         'color': COLORS['cal'], 'weight': 'bold'},
        {'x': 17.5, 'y': 0.35, 'text': 'W5–W8 (Test)', 'fs': 9,
         'color': COLORS['test'], 'weight': 'bold'},
        {'x': 17.5, 'y': 2.5, 'text': 'Same $\\hat{q}$ used for ALL test days → coverage degrades over time',
         'fs': 9, 'color': '#e15759', 'style': 'italic'},
    ],
    window_bracket=(1, 7),
)

# ============================================================
# Panel B: Online Expanding — calibration grows daily
# ============================================================
# At day 15 (middle of test), cal includes days 1-14
online_exp_cal = list(range(1, 16))  # days 1-15 in calibration
online_exp_test = list(range(16, 26))  # remaining test

draw_timeline(
    axes[1],
    '(b) Online Expanding Window — Calibration Grows Daily',
    cal_blocks=online_exp_cal,
    test_blocks=online_exp_test,
    annotations=[
        {'x': 8.5, 'y': 0.35, 'text': 'Original cal + observed test days', 'fs': 8,
         'color': COLORS['cal'], 'weight': 'normal'},
        {'x': 21.0, 'y': 0.35, 'text': 'Remaining test', 'fs': 8,
         'color': COLORS['test'], 'weight': 'normal'},
        {'x': 17.5, 'y': 2.5,
         'text': '$\\hat{q}$ recalculated daily with growing calibration set → adapts to drift',
         'fs': 9, 'color': '#59a14f', 'style': 'italic'},
    ],
    window_bracket=(1, 15),
)

# Add "grows" arrow
axes[1].annotate('', xy=(16.5, 2.15), xytext=(8.0, 2.15),
                 arrowprops=dict(arrowstyle='->', color=COLORS['cal'],
                                 lw=2, mutation_scale=15))
axes[1].text(12.0, 2.3, 'grows daily', ha='center', va='bottom',
             fontsize=8, color=COLORS['cal'], fontweight='bold')

# ============================================================
# Panel C: Online Sliding Window — fixed-size moving window
# ============================================================
window_size = 7  # 7-day sliding window
current_day = 18
slide_start = current_day - window_size + 1
slide_end = current_day

slide_cal = list(range(slide_start, slide_end + 1))
slide_removed = list(range(1, slide_start))
slide_test = list(range(slide_end + 1, 26))

draw_timeline(
    axes[2],
    '(c) Online Sliding Window (7-day) — Fixed-Size Moving Window',
    cal_blocks=slide_cal,
    test_blocks=slide_test,
    removed_blocks=slide_removed,
    annotations=[
        {'x': 5.0, 'y': 0.35, 'text': 'Dropped (too old)', 'fs': 8,
         'color': COLORS['removed'], 'weight': 'normal'},
        {'x': 15.5, 'y': 0.35, 'text': '7-day window', 'fs': 8,
         'color': COLORS['cal'], 'weight': 'bold'},
        {'x': 22.5, 'y': 0.35, 'text': 'Remaining test', 'fs': 8,
         'color': COLORS['test'], 'weight': 'normal'},
        {'x': 17.5, 'y': 2.5,
         'text': 'Window slides forward → only recent data used → maximizes recency',
         'fs': 9, 'color': '#b07aa1', 'style': 'italic'},
    ],
    window_bracket=(slide_start, slide_end),
)

# Slide arrow
axes[2].annotate('', xy=(slide_end * 1.0 + 1.5, 2.15),
                 xytext=(slide_start * 1.0, 2.15),
                 arrowprops=dict(arrowstyle='->', color=COLORS['window'],
                                 lw=2, mutation_scale=15))
axes[2].text((slide_start + slide_end) / 2 * 1.0 + 0.5, 2.3,
             'slides forward', ha='center', va='bottom',
             fontsize=8, color=COLORS['window'], fontweight='bold')

# ============================================================
# SHARED LEGEND
# ============================================================
legend_items = [
    mpatches.Patch(facecolor=COLORS['cal'], edgecolor='white', label='Calibration data'),
    mpatches.Patch(facecolor=COLORS['test'], edgecolor='white', label='Test data (to predict)'),
    mpatches.Patch(facecolor=COLORS['stale'], edgecolor='white', alpha=0.4, label='Stale / unused'),
    mpatches.Patch(facecolor=COLORS['removed'], edgecolor='white', alpha=0.5, label='Dropped by sliding window'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=4, fontsize=9,
           frameon=True, edgecolor='#dddddd', fancybox=True,
           bbox_to_anchor=(0.5, -0.02))

plt.subplots_adjust(hspace=0.4)

# ============================================================
# SAVE
# ============================================================
output_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(output_dir, 'fig_3_6_online_vs_static.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(output_dir, 'fig_3_6_online_vs_static.pdf'),
            bbox_inches='tight', facecolor='white')
print("Saved fig_3_6")
plt.close()