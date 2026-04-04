"""
Generate Figure 3.4: Data Preprocessing Pipeline
Shows each stage with record counts and removal rates.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

# ============================================================
# PIPELINE STAGES
# ============================================================
stages = [
    {
        'label': 'Raw Dataset',
        'records': '785,976',
        'detail': '19,769 trips\n55 days, 3 routes',
        'color': '#4e79a7',
        'removed': None,
    },
    {
        'label': '1. Duplicate\nRemoval',
        'records': '785,976',
        'detail': 'Exact duplicate\nrow detection',
        'color': '#76b7b2',
        'removed': '0 removed (0%)',
        'removed_color': '#59a14f',
    },
    {
        'label': '2. Anomalous\nDate Filtering',
        'records': '782,719',
        'detail': 'Sep 3-4 excluded\n(min 5,000 records/day)',
        'color': '#76b7b2',
        'removed': '3,257 removed (0.41%)',
        'removed_color': '#f28e2b',
    },
    {
        'label': '3. IQR Outlier\nRemoval',
        'records': '749,810',
        'detail': 'Per segment + direction\nthreshold = 1.5 x IQR',
        'color': '#76b7b2',
        'removed': '32,909 removed (4.20%)',
        'removed_color': '#e15759',
    },
    {
        'label': '4. Trip\nCompleteness\nFiltering',
        'records': '747,798',
        'detail': 'Min 30 segments/trip\n71 trips excluded',
        'color': '#76b7b2',
        'removed': '2,012 removed (0.27%)',
        'removed_color': '#f28e2b',
    },
    {
        'label': '5. Travel Time\nComputation &\nAggregation',
        'records': '747,798',
        'detail': 'Segment: run + dwell\nRoute: sum of segments',
        'color': '#59a14f',
        'removed': None,
    },
]

# Layout
n = len(stages)
box_w = 1.7
box_h = 1.3
gap = 0.6
total_w = n * box_w + (n - 1) * gap
x_start = (14 - total_w) / 2
y_main = 4.0

for i, s in enumerate(stages):
    x = x_start + i * (box_w + gap)

    # Main box
    fancy = FancyBboxPatch(
        (x, y_main), box_w, box_h,
        boxstyle="round,pad=0.1",
        facecolor=s['color'], edgecolor='white', linewidth=2, alpha=0.9
    )
    ax.add_patch(fancy)
    ax.text(x + box_w / 2, y_main + box_h / 2, s['label'],
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', linespacing=1.3)

    # Record count below
    ax.text(x + box_w / 2, y_main - 0.2, s['records'] + ' records',
            ha='center', va='top', fontsize=8, fontweight='bold',
            color='#333333')

    # Detail box below records
    detail_y = y_main - 0.95
    detail_box = FancyBboxPatch(
        (x - 0.05, detail_y - 0.15), box_w + 0.1, 0.65,
        boxstyle="round,pad=0.06",
        facecolor='#f7f7f7', edgecolor='#dddddd', linewidth=1
    )
    ax.add_patch(detail_box)
    ax.text(x + box_w / 2, detail_y + 0.15, s['detail'],
            ha='center', va='center', fontsize=7, color='#555555',
            linespacing=1.2)

    # Removed count (above, in red/orange badge)
    if s['removed'] is not None:
        badge_y = y_main + box_h + 0.15
        badge_box = FancyBboxPatch(
            (x - 0.1, badge_y), box_w + 0.2, 0.4,
            boxstyle="round,pad=0.06",
            facecolor=s['removed_color'], edgecolor='white',
            linewidth=1.5, alpha=0.85
        )
        ax.add_patch(badge_box)
        ax.text(x + box_w / 2, badge_y + 0.2, s['removed'],
                ha='center', va='center', fontsize=7.5, fontweight='bold',
                color='white')

    # Arrow to next
    if i < n - 1:
        ax.annotate('',
                    xy=(x + box_w + gap * 0.15, y_main + box_h / 2),
                    xytext=(x + box_w + 0.05, y_main + box_h / 2),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                   lw=2, mutation_scale=15))

# ============================================================
# OUTPUT BOXES (bottom)
# ============================================================
output_y = 1.2
out_w = 4.5
out_h = 0.9

# Segment-level output
seg_x = 2.5
seg_box = FancyBboxPatch(
    (seg_x, output_y), out_w, out_h,
    boxstyle="round,pad=0.1",
    facecolor='#4e79a7', edgecolor='white', linewidth=2, alpha=0.85
)
ax.add_patch(seg_box)
ax.text(seg_x + out_w / 2, output_y + out_h / 2 + 0.15,
        'Segment-Level Dataset', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')
ax.text(seg_x + out_w / 2, output_y + out_h / 2 - 0.2,
        '747,798 records  |  18 columns  |  temporal labels',
        ha='center', va='center', fontsize=7.5, color='#ddeeff')

# Route-level output
route_x = 7.5
route_box = FancyBboxPatch(
    (route_x, output_y), out_w, out_h,
    boxstyle="round,pad=0.1",
    facecolor='#59a14f', edgecolor='white', linewidth=2, alpha=0.85
)
ax.add_patch(route_box)
ax.text(route_x + out_w / 2, output_y + out_h / 2 + 0.15,
        'Route-Level Dataset', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')
ax.text(route_x + out_w / 2, output_y + out_h / 2 - 0.2,
        '19,614 trips  |  11 columns  |  temporal labels',
        ha='center', va='center', fontsize=7.5, color='#ddeeff')

# Arrows from last stage down to outputs
last_x = x_start + (n - 1) * (box_w + gap) + box_w / 2
fork_y = y_main - 1.3

ax.plot([last_x, last_x], [fork_y, output_y + out_h + 0.4],
        color='#888888', linewidth=1.5, linestyle='--')
ax.plot([last_x, seg_x + out_w / 2], [output_y + out_h + 0.4, output_y + out_h + 0.4],
        color='#888888', linewidth=1.5, linestyle='--')
ax.plot([last_x, route_x + out_w / 2], [output_y + out_h + 0.4, output_y + out_h + 0.4],
        color='#888888', linewidth=1.5, linestyle='--')

ax.annotate('', xy=(seg_x + out_w / 2, output_y + out_h),
            xytext=(seg_x + out_w / 2, output_y + out_h + 0.4),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))
ax.annotate('', xy=(route_x + out_w / 2, output_y + out_h),
            xytext=(route_x + out_w / 2, output_y + out_h + 0.4),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))

# ============================================================
# TOTAL SUMMARY (top-right corner)
# ============================================================
summary_x = 10.5
summary_y = 6.3
summary_box = FancyBboxPatch(
    (summary_x, summary_y - 0.3), 3.2, 0.6,
    boxstyle="round,pad=0.1",
    facecolor='#f0f0f0', edgecolor='#cccccc', linewidth=1
)
ax.add_patch(summary_box)
ax.text(summary_x + 1.6, summary_y, 'Total removed: 38,178 records (4.86%)',
        ha='center', va='center', fontsize=8.5, fontweight='bold',
        color='#e15759')

# ============================================================
# SAVE
# ============================================================
output_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(output_dir, 'fig_3_4_preprocessing_pipeline.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(output_dir, 'fig_3_4_preprocessing_pipeline.pdf'),
            bbox_inches='tight', facecolor='white')
print(f"Saved to {output_dir}/fig_3_4_preprocessing_pipeline.png and .pdf")
plt.close()