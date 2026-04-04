"""
Figure 3.5: How Split Conformal Prediction Works
Shows: Train model → Compute residuals on calibration → Take quantile → Form intervals
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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

fig = plt.figure(figsize=(15, 8.5))

# Two-row layout: top = conceptual pipeline, bottom = visual example
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.35)

# ============================================================
# TOP: Conceptual Pipeline (4 steps)
# ============================================================
ax_top = fig.add_subplot(gs[0])
ax_top.set_xlim(0, 15)
ax_top.set_ylim(0, 3.5)
ax_top.axis('off')

steps = [
    {
        'label': 'Step 1\nTrain Model',
        'detail': 'XGBoost trained\non W1–W3\n(7,598 trips)',
        'color': '#4e79a7',
        'x': 0.3,
    },
    {
        'label': 'Step 2\nCompute Residuals',
        'detail': 'Predict on W4\n$R_i = |y_i - \\hat{y}_i|$\n(2,740 scores)',
        'color': '#76b7b2',
        'x': 3.3,
    },
    {
        'label': 'Step 3\nFind Quantile',
        'detail': 'Sort residuals\n$\\hat{q}$ = 90th percentile\n(threshold)',
        'color': '#f28e2b',
        'x': 6.3,
    },
    {
        'label': 'Step 4\nForm Intervals',
        'detail': 'For each test sample:\n$[\\hat{y} - \\hat{q},\\; \\hat{y} + \\hat{q}]$\n(constant width)',
        'color': '#e15759',
        'x': 9.3,
    },
]

bw = 2.5
bh = 2.8

for i, s in enumerate(steps):
    x = s['x']
    box = FancyBboxPatch(
        (x, 0.3), bw, bh,
        boxstyle="round,pad=0.12",
        facecolor=s['color'], edgecolor='white', linewidth=2, alpha=0.9
    )
    ax_top.add_patch(box)

    ax_top.text(x + bw / 2, 0.3 + bh - 0.45, s['label'],
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white', linespacing=1.3)

    ax_top.text(x + bw / 2, 0.3 + bh / 2 - 0.35, s['detail'],
                ha='center', va='center', fontsize=9, color='white',
                linespacing=1.3)

    if i < len(steps) - 1:
        ax_top.annotate('',
                        xy=(steps[i + 1]['x'] - 0.1, 0.3 + bh / 2),
                        xytext=(x + bw + 0.1, 0.3 + bh / 2),
                        arrowprops=dict(arrowstyle='->', color='#444444',
                                        lw=2.5, mutation_scale=18))

# Output box
out_x = 12.3
out_box = FancyBboxPatch(
    (out_x, 0.7), 2.3, 2.0,
    boxstyle="round,pad=0.12",
    facecolor='#59a14f', edgecolor='white', linewidth=2, alpha=0.9
)
ax_top.add_patch(out_box)
ax_top.text(out_x + 1.15, 1.7 + 0.45, 'Output', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
ax_top.text(out_x + 1.15, 1.7 - 0.25, 'Prediction\nIntervals\nwith 90%\ncoverage\nguarantee',
            ha='center', va='center', fontsize=8.5, color='white', linespacing=1.2)

ax_top.annotate('',
                xy=(out_x - 0.1, 1.7),
                xytext=(steps[-1]['x'] + bw + 0.1, 1.7),
                arrowprops=dict(arrowstyle='->', color='#444444',
                                lw=2.5, mutation_scale=18))

# ============================================================
# BOTTOM: Visual Example
# ============================================================
ax_bot = fig.add_subplot(gs[1])

np.random.seed(42)

# Simulate calibration residuals
n_cal = 80
residuals = np.abs(np.random.normal(0, 400, n_cal)) + np.random.exponential(200, n_cal)
residuals = np.sort(residuals)
q_90 = np.percentile(residuals, 90)

# Simulate test predictions
n_test = 25
x_test = np.arange(n_test)
y_pred = np.random.uniform(3500, 6500, n_test)
y_true = y_pred + np.random.normal(0, 500, n_test)

lower = y_pred - q_90
upper = y_pred + q_90
covered = (y_true >= lower) & (y_true <= upper)

# Left panel: Residual histogram
ax_hist = fig.add_axes([0.06, 0.05, 0.28, 0.42])
ax_hist.hist(residuals, bins=20, color='#76b7b2', edgecolor='white',
             linewidth=0.8, alpha=0.85)
ax_hist.axvline(q_90, color='#e15759', linewidth=2.5, linestyle='--',
                label=f'$\\hat{{q}}$ = {q_90:.0f}s (90th percentile)')
ax_hist.fill_betweenx([0, ax_hist.get_ylim()[1] if ax_hist.get_ylim()[1] > 0 else 15],
                       q_90, residuals.max() + 50,
                       alpha=0.15, color='#e15759')

ax_hist.set_xlabel('Nonconformity Score $R_i = |y_i - \\hat{y}_i|$ (seconds)', fontsize=9)
ax_hist.set_ylabel('Frequency', fontsize=9)
ax_hist.set_title('Calibration Residuals (W4)', fontsize=10, fontweight='bold')
ax_hist.legend(fontsize=8, loc='upper right')
ax_hist.spines['top'].set_visible(False)
ax_hist.spines['right'].set_visible(False)
ax_hist.tick_params(labelsize=8)

# Fix histogram y-axis for fill
ylim = ax_hist.get_ylim()
ax_hist.fill_betweenx([0, ylim[1]], q_90, residuals.max() + 50,
                       alpha=0.1, color='#e15759')

# Right panel: Prediction intervals on test data
ax_pred = fig.add_axes([0.42, 0.05, 0.55, 0.42])

# Draw intervals
for i in range(n_test):
    color = '#59a14f' if covered[i] else '#e15759'
    ax_pred.plot([x_test[i], x_test[i]], [lower[i], upper[i]],
                 color=color, linewidth=2, alpha=0.5)

# Draw predictions and true values
ax_pred.scatter(x_test, y_pred, color='#4e79a7', s=40, zorder=5,
                label='Point prediction $\\hat{y}$', marker='s')
ax_pred.scatter(x_test[covered], y_true[covered], color='#59a14f', s=45,
                zorder=6, label=f'True value (covered)', marker='o', edgecolors='white', linewidth=0.5)
ax_pred.scatter(x_test[~covered], y_true[~covered], color='#e15759', s=55,
                zorder=6, label=f'True value (not covered)', marker='X', edgecolors='white', linewidth=0.5)

# Annotate interval width
mid_idx = 12
ax_pred.annotate('', xy=(n_test + 0.8, upper[mid_idx]),
                 xytext=(n_test + 0.8, lower[mid_idx]),
                 arrowprops=dict(arrowstyle='<->', color='#f28e2b', lw=2))
ax_pred.text(n_test + 1.3, y_pred[mid_idx],
             f'Width = 2$\\hat{{q}}$\n= {2 * q_90:.0f}s\n({2 * q_90 / 60:.0f} min)',
             ha='left', va='center', fontsize=8, color='#f28e2b', fontweight='bold')

picp = covered.sum() / n_test
ax_pred.set_xlabel('Test Sample Index', fontsize=9)
ax_pred.set_ylabel('Travel Time (seconds)', fontsize=9)
ax_pred.set_title(f'Prediction Intervals on Test Data  (PICP = {picp:.0%})',
                  fontsize=10, fontweight='bold')
ax_pred.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax_pred.spines['top'].set_visible(False)
ax_pred.spines['right'].set_visible(False)
ax_pred.tick_params(labelsize=8)

# Hide the background axes
ax_bot.axis('off')

# ============================================================
# SAVE
# ============================================================
output_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(output_dir, 'fig_3_5_cp_visualization.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(output_dir, 'fig_3_5_cp_visualization.pdf'),
            bbox_inches='tight', facecolor='white')
print(f"Saved fig_3_5")
plt.close()