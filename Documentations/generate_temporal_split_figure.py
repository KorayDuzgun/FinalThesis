"""
Generate Figure 3.2: Temporal Split Strategy
Training (W1-W3), Calibration (W4), Test-Near (W5), Test-Mid (W6), Test-Far (W7-W8)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.dates as mdates
from datetime import datetime, timedelta
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

# --- Period colors (matching project palette) ---
COLORS = {
    'train':       '#4e79a7',
    'calibration': '#59a14f',
    'test_near':   '#f28e2b',
    'test_mid':    '#e15759',
    'test_far':    '#b07aa1',
    'excluded':    '#bab0ac',
}

# --- Week boundaries ---
weeks = [
    {'label': 'W1', 'start': datetime(2024, 7, 29), 'end': datetime(2024, 8, 4),  'period': 'train'},
    {'label': 'W2', 'start': datetime(2024, 8, 5),  'end': datetime(2024, 8, 11), 'period': 'train'},
    {'label': 'W3', 'start': datetime(2024, 8, 12), 'end': datetime(2024, 8, 18), 'period': 'train'},
    {'label': 'W4', 'start': datetime(2024, 8, 19), 'end': datetime(2024, 8, 25), 'period': 'calibration'},
    {'label': 'W5', 'start': datetime(2024, 8, 26), 'end': datetime(2024, 9, 1),  'period': 'test_near'},
    {'label': 'W6', 'start': datetime(2024, 9, 2),  'end': datetime(2024, 9, 8),  'period': 'test_mid'},
    {'label': 'W7', 'start': datetime(2024, 9, 9),  'end': datetime(2024, 9, 15), 'period': 'test_far'},
    {'label': 'W8', 'start': datetime(2024, 9, 16), 'end': datetime(2024, 9, 21), 'period': 'test_far'},
]

# Anomalous dates
anomalous = [datetime(2024, 9, 3), datetime(2024, 9, 4)]

fig, ax = plt.subplots(figsize=(14, 5.5))

# ============================================================
# MAIN TIMELINE BAR (top)
# ============================================================
bar_y = 3.0
bar_height = 1.0
total_start = datetime(2024, 7, 29)
total_end = datetime(2024, 9, 21)
total_days = (total_end - total_start).days

def date_to_x(d):
    return 0.5 + 13.0 * (d - total_start).days / total_days

# Draw week blocks
for w in weeks:
    x_start = date_to_x(w['start'])
    x_end = date_to_x(w['end'] + timedelta(days=1))
    width = x_end - x_start
    color = COLORS[w['period']]

    rect = FancyBboxPatch(
        (x_start, bar_y), width - 0.02, bar_height,
        boxstyle="round,pad=0.03",
        facecolor=color, edgecolor='white', linewidth=2, alpha=0.9
    )
    ax.add_patch(rect)

    # Week label
    ax.text(x_start + width/2, bar_y + bar_height/2, w['label'],
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Mark anomalous dates with X pattern
for ad in anomalous:
    x = date_to_x(ad)
    x_end = date_to_x(ad + timedelta(days=1))
    w = x_end - x
    rect = plt.Rectangle((x, bar_y), w - 0.02, bar_height,
                          facecolor=COLORS['excluded'], edgecolor='white',
                          linewidth=1, alpha=0.95, zorder=5)
    ax.add_patch(rect)
    ax.text(x + w/2, bar_y + bar_height/2, 'X',
            ha='center', va='center', fontsize=13, color='white',
            fontweight='bold', zorder=6)

# ============================================================
# DATE LABELS (below the bar)
# ============================================================
date_y = 2.55
for w in weeks:
    x_start = date_to_x(w['start'])
    x_end = date_to_x(w['end'] + timedelta(days=1))
    mid = (x_start + x_end) / 2

    start_str = w['start'].strftime('%b %d')
    end_str = w['end'].strftime('%b %d')
    ax.text(mid, date_y, f"{start_str}–{end_str}",
            ha='center', va='top', fontsize=7, color='#666666')

# ============================================================
# PERIOD GROUP BRACKETS (above the bar)
# ============================================================
bracket_y = 4.25
label_y = 4.7

period_groups = [
    {'name': 'Training', 'weeks': ['W1', 'W2', 'W3'], 'color': COLORS['train'],
     'trips': '7,598 trips', 'days': '21 days'},
    {'name': 'Calibration', 'weeks': ['W4'], 'color': COLORS['calibration'],
     'trips': '2,740 trips', 'days': '7 days'},
    {'name': 'Test-Near', 'weeks': ['W5'], 'color': COLORS['test_near'],
     'trips': '2,707 trips', 'days': '7 days'},
    {'name': 'Test-Mid', 'weeks': ['W6'], 'color': COLORS['test_mid'],
     'trips': '1,833 trips', 'days': '5 days*'},
    {'name': 'Test-Far', 'weeks': ['W7', 'W8'], 'color': COLORS['test_far'],
     'trips': '4,736 trips', 'days': '13 days'},
]

for pg in period_groups:
    # Find x range for this group
    group_weeks = [w for w in weeks if w['label'] in pg['weeks']]
    x_left = date_to_x(group_weeks[0]['start'])
    x_right = date_to_x(group_weeks[-1]['end'] + timedelta(days=1)) - 0.02
    mid = (x_left + x_right) / 2

    # Bracket line
    ax.plot([x_left, x_left], [bar_y + bar_height + 0.05, bracket_y],
            color=pg['color'], linewidth=1.5)
    ax.plot([x_right, x_right], [bar_y + bar_height + 0.05, bracket_y],
            color=pg['color'], linewidth=1.5)
    ax.plot([x_left, x_right], [bracket_y, bracket_y],
            color=pg['color'], linewidth=1.5)

    # Period name
    ax.text(mid, label_y + 0.15, pg['name'],
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color=pg['color'])

    # Trip count
    ax.text(mid, label_y - 0.15, f"{pg['trips']} ({pg['days']})",
            ha='center', va='top', fontsize=8, color='#666666')

# ============================================================
# TEMPORAL DISTANCE ARROWS (bottom section)
# ============================================================
arrow_y = 1.6

# Reference point: end of calibration
cal_end_x = date_to_x(datetime(2024, 8, 26))

# Draw base line
ax.plot([cal_end_x, date_to_x(total_end + timedelta(days=1))],
        [arrow_y, arrow_y], color='#cccccc', linewidth=1, linestyle='-')

# Distance markers
distances = [
    {'label': '1–7 days', 'period': 'test_near',
     'x_end': date_to_x(datetime(2024, 9, 2))},
    {'label': '8–14 days', 'period': 'test_mid',
     'x_end': date_to_x(datetime(2024, 9, 9))},
    {'label': '15–27 days', 'period': 'test_far',
     'x_end': date_to_x(datetime(2024, 9, 22))},
]

for i, dist in enumerate(distances):
    color = COLORS[dist['period']]
    y_off = -0.35 * i

    # Arrow
    ax.annotate('', xy=(dist['x_end'], arrow_y + y_off),
                xytext=(cal_end_x, arrow_y + y_off),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.8))

    # Label
    mid = (cal_end_x + dist['x_end']) / 2
    ax.text(mid, arrow_y + y_off + 0.15, dist['label'],
            ha='center', va='bottom', fontsize=8, fontweight='bold',
            color=color)

# "Temporal distance from calibration" label
ax.text(cal_end_x - 0.15, arrow_y - 0.35, "Temporal distance\nfrom calibration →",
        ha='right', va='center', fontsize=8, fontstyle='italic', color='#888888')

# ============================================================
# ANNOTATIONS
# ============================================================
# Anomalous dates note
ax.text(date_to_x(datetime(2024, 9, 3)) + 0.15, bar_y - 0.15,
        "* Sep 3–4 excluded\n  (data collection failure)",
        ha='left', va='top', fontsize=7, color=COLORS['excluded'],
        fontstyle='italic')

# Increasing drift arrow at very bottom
drift_x_start = date_to_x(datetime(2024, 8, 26))
drift_x_end = date_to_x(datetime(2024, 9, 22))
drift_y = 0.35

# Gradient bar for "increasing drift"
n_grad = 100
for j in range(n_grad):
    frac = j / n_grad
    x = drift_x_start + frac * (drift_x_end - drift_x_start)
    w = (drift_x_end - drift_x_start) / n_grad
    r = 0.31 + 0.69 * frac  # from blue-ish to red-ish
    g = 0.47 - 0.25 * frac
    b = 0.65 - 0.30 * frac
    ax.add_patch(plt.Rectangle((x, drift_y), w, 0.25,
                                facecolor=(r, g, b, 0.6), edgecolor='none'))

ax.text((drift_x_start + drift_x_end) / 2, drift_y + 0.12,
        "Increasing Distribution Shift →",
        ha='center', va='center', fontsize=9, fontweight='bold',
        color='white')

# ============================================================
# LEGEND
# ============================================================
legend_y = 0.35
legend_items = [
    ('Training (W1–W3)', COLORS['train']),
    ('Calibration (W4)', COLORS['calibration']),
    ('Test-Near (W5)', COLORS['test_near']),
    ('Test-Mid (W6)', COLORS['test_mid']),
    ('Test-Far (W7–W8)', COLORS['test_far']),
    ('Excluded', COLORS['excluded']),
]

legend_patches = [mpatches.Patch(facecolor=c, edgecolor='white', label=l)
                  for l, c in legend_items]
ax.legend(handles=legend_patches, loc='lower left',
          bbox_to_anchor=(0.0, -0.08), ncol=6, fontsize=8,
          frameon=True, edgecolor='#dddddd', fancybox=True)

# ============================================================
# FINAL STYLING
# ============================================================
ax.set_xlim(0, 14)
ax.set_ylim(-0.1, 5.4)
ax.axis('off')

# ============================================================
# SAVE
# ============================================================
output_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(output_dir, 'fig_3_2_temporal_split.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(output_dir, 'fig_3_2_temporal_split.pdf'),
            bbox_inches='tight', facecolor='white')
print(f"Saved to {output_dir}/fig_3_2_temporal_split.png and .pdf")
plt.close()