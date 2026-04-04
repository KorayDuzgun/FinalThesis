"""
Figure 3.7: Segment-Level Uncertainty Decomposition
Shows: Route split into segments → per-segment CP → aggregation → attribution
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

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# ============================================================
# TOP: Route-level bar (one big prediction)
# ============================================================
ax.text(7.0, 8.6, 'Route-Level vs. Segment-Level Conformal Prediction',
        ha='center', va='center', fontsize=13, fontweight='bold', color='#333333')

# Route bar
route_y = 7.5
route_x = 1.0
route_w = 12.0
route_h = 0.7

route_box = FancyBboxPatch(
    (route_x, route_y), route_w, route_h,
    boxstyle="round,pad=0.06",
    facecolor='#4e79a7', edgecolor='white', linewidth=2, alpha=0.9
)
ax.add_patch(route_box)
ax.text(route_x + route_w / 2, route_y + route_h / 2,
        'Route-Level: Single prediction interval for entire trip    '
        '$[\\hat{y}_{route} - \\hat{q},\\;\\hat{y}_{route} + \\hat{q}]$',
        ha='center', va='center', fontsize=10, color='white', fontweight='bold')

ax.text(route_x + route_w + 0.15, route_y + route_h / 2,
        '← constant\n   width',
        ha='left', va='center', fontsize=8, color='#e15759', fontstyle='italic')

# ============================================================
# MIDDLE: Segment decomposition
# ============================================================
ax.text(7.0, 6.8, '▼  Decompose into segments  ▼',
        ha='center', va='center', fontsize=10, color='#888888')

seg_y = 5.6
n_segments = 8
seg_gap = 0.1
total_seg_w = route_w - (n_segments - 1) * seg_gap
seg_widths_raw = np.array([1.2, 0.8, 1.5, 1.0, 2.0, 0.9, 1.3, 1.1])
seg_widths = seg_widths_raw / seg_widths_raw.sum() * total_seg_w
seg_h = 0.7

# Colors: intensity based on uncertainty (wider = more red)
uncertainty = np.array([0.3, 0.15, 0.6, 0.25, 0.9, 0.2, 0.5, 0.35])
uncertainty_norm = uncertainty / uncertainty.max()

seg_labels = [f'S{i+1}' for i in range(n_segments)]

x_cursor = route_x
segment_positions = []

for i in range(n_segments):
    w = seg_widths[i]
    u = uncertainty_norm[i]

    # Color gradient: green (low uncertainty) to red (high uncertainty)
    r = 0.31 + 0.58 * u
    g = 0.60 - 0.35 * u
    b = 0.47 - 0.20 * u

    rect = FancyBboxPatch(
        (x_cursor, seg_y), w, seg_h,
        boxstyle="round,pad=0.04",
        facecolor=(r, g, b, 0.9), edgecolor='white', linewidth=1.5
    )
    ax.add_patch(rect)

    ax.text(x_cursor + w / 2, seg_y + seg_h / 2, seg_labels[i],
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    segment_positions.append((x_cursor, w))
    x_cursor += w + seg_gap

# ============================================================
# SEGMENT-LEVEL INTERVALS (below segments)
# ============================================================
ax.text(7.0, 5.15, 'Per-segment conformal prediction  →  adaptive widths ($\\hat{q}_s$ varies by segment)',
        ha='center', va='center', fontsize=9, color='#666666', fontstyle='italic')

interval_y = 4.0
interval_h = 0.8

for i in range(n_segments):
    sx, sw = segment_positions[i]
    u = uncertainty_norm[i]

    # Interval width proportional to uncertainty
    int_w = sw * (0.6 + 0.8 * u)
    int_x = sx + sw / 2 - int_w / 2

    r = 0.31 + 0.58 * u
    g = 0.60 - 0.35 * u
    b = 0.47 - 0.20 * u

    # Interval box
    rect = Rectangle(
        (int_x, interval_y), int_w, interval_h,
        facecolor=(r, g, b, 0.25), edgecolor=(r, g, b, 0.9), linewidth=1.5
    )
    ax.add_patch(rect)

    # Point prediction dot
    ax.plot(sx + sw / 2, interval_y + interval_h / 2, 'o',
            color=(r, g, b, 0.9), markersize=5, zorder=5)

    # Width label for notable segments
    if i == 4:  # Widest (S5)
        ax.annotate('Wide\n(high $\\hat{q}_5$)',
                    xy=(sx + sw / 2, interval_y - 0.05),
                    ha='center', va='top', fontsize=7, color='#e15759',
                    fontweight='bold')
    elif i == 1:  # Narrowest (S2)
        ax.annotate('Narrow\n(low $\\hat{q}_2$)',
                    xy=(sx + sw / 2, interval_y - 0.05),
                    ha='center', va='top', fontsize=7, color='#59a14f',
                    fontweight='bold')

    # Arrow from segment to interval
    ax.annotate('', xy=(sx + sw / 2, interval_y + interval_h),
                xytext=(sx + sw / 2, seg_y),
                arrowprops=dict(arrowstyle='->', color='#cccccc', lw=0.8))

# ============================================================
# AGGREGATION (bottom-left)
# ============================================================
agg_y = 1.8
agg_x = 0.5
agg_w = 6.0
agg_h = 1.5

agg_box = FancyBboxPatch(
    (agg_x, agg_y), agg_w, agg_h,
    boxstyle="round,pad=0.12",
    facecolor='#4e79a7', edgecolor='white', linewidth=2, alpha=0.85
)
ax.add_patch(agg_box)
ax.text(agg_x + agg_w / 2, agg_y + agg_h - 0.35,
        'Route-Level Aggregation', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(agg_x + agg_w / 2, agg_y + agg_h / 2 - 0.2,
        '$\\ell_{route} = \\sum_{s=1}^{N} \\ell_s$'
        '  ,   $u_{route} = \\sum_{s=1}^{N} u_s$',
        ha='center', va='center', fontsize=10, color='white')
ax.text(agg_x + agg_w / 2, agg_y + 0.25,
        'Coverage preserved (PICP ≥ 90%)',
        ha='center', va='center', fontsize=8.5, color='#c8e6c9')

# Arrow from intervals to aggregation
ax.annotate('', xy=(agg_x + agg_w / 2, agg_y + agg_h),
            xytext=(agg_x + agg_w / 2, interval_y),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=2,
                            connectionstyle='arc3,rad=0'))

# ============================================================
# ATTRIBUTION (bottom-right)
# ============================================================
attr_x = 7.5
attr_w = 6.0
attr_h = 1.5

attr_box = FancyBboxPatch(
    (attr_x, agg_y), attr_w, attr_h,
    boxstyle="round,pad=0.12",
    facecolor='#b07aa1', edgecolor='white', linewidth=2, alpha=0.85
)
ax.add_patch(attr_box)
ax.text(attr_x + attr_w / 2, agg_y + attr_h - 0.35,
        'Uncertainty Attribution', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(attr_x + attr_w / 2, agg_y + attr_h / 2 - 0.2,
        '$f_s = \\frac{u_s - \\ell_s}{\\sum_{s\'} (u_{s\'} - \\ell_{s\'})}$'
        '     (fraction per segment)',
        ha='center', va='center', fontsize=10, color='white')
ax.text(attr_x + attr_w / 2, agg_y + 0.25,
        'Identifies uncertainty hotspots',
        ha='center', va='center', fontsize=8.5, color='#e1bee7')

# Arrow from intervals to attribution
ax.annotate('', xy=(attr_x + attr_w / 2, agg_y + attr_h),
            xytext=(attr_x + attr_w / 2, interval_y),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=2,
                            connectionstyle='arc3,rad=0'))

# ============================================================
# EXAMPLE BAR CHART (bottom, uncertainty fractions)
# ============================================================
ax_bar = fig.add_axes([0.57, 0.02, 0.38, 0.17])
fractions = uncertainty / uncertainty.sum()
colors_bar = []
for u in uncertainty_norm:
    r = 0.31 + 0.58 * u
    g = 0.60 - 0.35 * u
    b = 0.47 - 0.20 * u
    colors_bar.append((r, g, b))

bars = ax_bar.bar(seg_labels, fractions * 100, color=colors_bar, edgecolor='white', linewidth=0.8)
ax_bar.set_ylabel('Contribution (%)', fontsize=8)
ax_bar.set_title('Segment Uncertainty Fraction ($f_s$)', fontsize=9, fontweight='bold')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.tick_params(labelsize=7)

# Highlight top segment
max_idx = np.argmax(fractions)
bars[max_idx].set_edgecolor('#e15759')
bars[max_idx].set_linewidth(2)
ax_bar.text(max_idx, fractions[max_idx] * 100 + 1,
            f'{fractions[max_idx]:.0%}', ha='center', fontsize=7,
            fontweight='bold', color='#e15759')

# ============================================================
# COLOR LEGEND
# ============================================================
ax.text(0.5, 0.7, 'Segment color intensity = uncertainty level:',
        ha='left', va='center', fontsize=8, color='#666666')

# Gradient bar
n_grad = 100
for j in range(n_grad):
    frac = j / n_grad
    r = 0.31 + 0.58 * frac
    g = 0.60 - 0.35 * frac
    b = 0.47 - 0.20 * frac
    ax.add_patch(Rectangle((5.5 + j * 0.03, 0.55), 0.03, 0.3,
                            facecolor=(r, g, b), edgecolor='none'))
ax.text(5.4, 0.7, 'Low', ha='right', va='center', fontsize=7, color='#59a14f')
ax.text(8.7, 0.7, 'High', ha='left', va='center', fontsize=7, color='#e15759')

# ============================================================
# SAVE
# ============================================================
output_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(output_dir, 'fig_3_7_segment_decomposition.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(output_dir, 'fig_3_7_segment_decomposition.pdf'),
            bbox_inches='tight', facecolor='white')
print("Saved fig_3_7")
plt.close()