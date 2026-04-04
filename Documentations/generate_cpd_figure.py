"""
Figure 3.8: Conformal Predictive Distribution (CPD) Example
Shows the step function P(y < t) for a single test sample,
with operationally relevant threshold queries marked.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
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

np.random.seed(42)

# Simulate calibration residuals (signed: r_i = y_i - y_hat_i)
n_cal = 200
residuals = np.random.normal(0, 450, n_cal) + np.random.exponential(100, n_cal) - 100
residuals = np.sort(residuals)

# Simulate a single test sample
y_hat = 4800  # point prediction: 80 minutes
y_true = 5400  # actual: 90 minutes

# Build CPD: for each threshold t, P(y < t) = |{i : y_hat + r_i <= t}| / (n+1)
t_range = np.linspace(2500, 7500, 1000)
cpd_values = np.array([np.sum(y_hat + residuals <= t) / (n_cal + 1) for t in t_range])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1.3, 1]})

# ============================================================
# LEFT: CPD Step Function
# ============================================================
ax = axes[0]

# Plot the CPD curve
ax.plot(t_range / 60, cpd_values, color='#4e79a7', linewidth=2.5, label='CPD: $p(t) = P(y < t)$')

# Mark the point prediction
ax.axvline(y_hat / 60, color='#59a14f', linewidth=1.5, linestyle='--', alpha=0.8)
ax.text(y_hat / 60 + 0.5, 0.03, f'$\\hat{{y}}$ = {y_hat/60:.0f} min\n(point prediction)',
        fontsize=8.5, color='#59a14f', fontweight='bold', va='bottom')

# Mark the true value
ax.axvline(y_true / 60, color='#e15759', linewidth=1.5, linestyle=':', alpha=0.8)
ax.text(y_true / 60 + 0.5, 0.93, f'$y_{{true}}$ = {y_true/60:.0f} min',
        fontsize=8.5, color='#e15759', fontweight='bold', va='top')

# Query 1: P(y < 90 min) = P(y < 5400s)
t1 = 5400
p1 = np.sum(y_hat + residuals <= t1) / (n_cal + 1)
ax.plot([t1/60, t1/60], [0, p1], color='#f28e2b', linewidth=1.5, linestyle='-', alpha=0.7)
ax.plot([t_range[0]/60, t1/60], [p1, p1], color='#f28e2b', linewidth=1.5, linestyle='-', alpha=0.7)
ax.plot(t1/60, p1, 'o', color='#f28e2b', markersize=8, zorder=5)
ax.annotate(f'P(y < 90 min) = {p1:.2f}',
            xy=(t1/60, p1), xytext=(t1/60 + 5, p1 - 0.08),
            fontsize=9, fontweight='bold', color='#f28e2b',
            arrowprops=dict(arrowstyle='->', color='#f28e2b', lw=1.5))

# Query 2: P(y < 75 min)
t2 = 4500
p2 = np.sum(y_hat + residuals <= t2) / (n_cal + 1)
ax.plot([t2/60, t2/60], [0, p2], color='#b07aa1', linewidth=1.5, linestyle='-', alpha=0.7)
ax.plot([t_range[0]/60, t2/60], [p2, p2], color='#b07aa1', linewidth=1.5, linestyle='-', alpha=0.7)
ax.plot(t2/60, p2, 'o', color='#b07aa1', markersize=8, zorder=5)
ax.annotate(f'P(y < 75 min) = {p2:.2f}',
            xy=(t2/60, p2), xytext=(t2/60 - 12, p2 + 0.06),
            fontsize=9, fontweight='bold', color='#b07aa1',
            arrowprops=dict(arrowstyle='->', color='#b07aa1', lw=1.5))

# Query 3: P(y < 105 min)
t3 = 6300
p3 = np.sum(y_hat + residuals <= t3) / (n_cal + 1)
ax.plot([t3/60, t3/60], [0, p3], color='#76b7b2', linewidth=1.5, linestyle='-', alpha=0.7)
ax.plot([t_range[0]/60, t3/60], [p3, p3], color='#76b7b2', linewidth=1.5, linestyle='-', alpha=0.7)
ax.plot(t3/60, p3, 'o', color='#76b7b2', markersize=8, zorder=5)
ax.annotate(f'P(y < 105 min) = {p3:.2f}',
            xy=(t3/60, p3), xytext=(t3/60 + 3, p3 - 0.08),
            fontsize=9, fontweight='bold', color='#76b7b2',
            arrowprops=dict(arrowstyle='->', color='#76b7b2', lw=1.5))

# 90% prediction interval shading
q05 = y_hat + np.percentile(residuals, 5)
q95 = y_hat + np.percentile(residuals, 95)
ax.axvspan(q05/60, q95/60, alpha=0.1, color='#4e79a7')
ax.text((q05 + q95) / 2 / 60, 0.5, '90% PI',
        ha='center', va='center', fontsize=9, color='#4e79a7',
        alpha=0.6, fontweight='bold')

# Reference lines
ax.axhline(0.5, color='#cccccc', linewidth=0.8, linestyle='-', alpha=0.5)
ax.axhline(0.9, color='#cccccc', linewidth=0.8, linestyle='-', alpha=0.5)
ax.text(t_range[-1]/60 - 0.5, 0.505, '50%', fontsize=7, color='#aaaaaa', ha='right')
ax.text(t_range[-1]/60 - 0.5, 0.905, '90%', fontsize=7, color='#aaaaaa', ha='right')

ax.set_xlabel('Threshold $t$ (minutes)', fontsize=11)
ax.set_ylabel('$p(t) = P(y < t)$', fontsize=11)
ax.set_title('Conformal Predictive Distribution\nfor a Single Test Sample', fontsize=12, fontweight='bold')
ax.set_ylim(-0.02, 1.05)
ax.set_xlim(t_range[0]/60, t_range[-1]/60)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)

# ============================================================
# RIGHT: Operational Use Cases
# ============================================================
ax2 = axes[1]
ax2.axis('off')

# Title
ax2.text(0.5, 0.95, 'Operational Queries via CPD',
         ha='center', va='top', fontsize=12, fontweight='bold',
         color='#333333', transform=ax2.transAxes)

# Use case boxes
use_cases = [
    {
        'question': '"Will this trip arrive\nbefore 90 minutes?"',
        'answer': f'P(y < 90 min) = {p1:.0%}',
        'color': '#f28e2b',
        'y': 0.78,
    },
    {
        'question': '"What is the risk of\nexceeding 105 minutes?"',
        'answer': f'P(y > 105 min) = {1-p3:.0%}',
        'color': '#76b7b2',
        'y': 0.55,
    },
    {
        'question': '"Will delay exceed\n15 min beyond prediction?"',
        'answer': f'P(y > $\\hat{{y}}$ + 15 min) = {1 - np.sum(y_hat + residuals <= y_hat + 900) / (n_cal + 1):.0%}',
        'color': '#b07aa1',
        'y': 0.32,
    },
]

for uc in use_cases:
    # Question box
    box = FancyBboxPatch(
        (0.02, uc['y'] - 0.08), 0.96, 0.18,
        boxstyle="round,pad=0.02",
        facecolor=uc['color'], edgecolor='white',
        linewidth=2, alpha=0.15,
        transform=ax2.transAxes
    )
    ax2.add_patch(box)

    ax2.text(0.05, uc['y'] + 0.04, uc['question'],
             ha='left', va='center', fontsize=9, fontstyle='italic',
             color='#555555', transform=ax2.transAxes, linespacing=1.3)

    ax2.text(0.95, uc['y'] + 0.04, uc['answer'],
             ha='right', va='center', fontsize=11, fontweight='bold',
             color=uc['color'], transform=ax2.transAxes)

# Formula box at bottom
formula_y = 0.1
formula_box = FancyBboxPatch(
    (0.02, formula_y - 0.05), 0.96, 0.12,
    boxstyle="round,pad=0.02",
    facecolor='#f0f0f0', edgecolor='#cccccc',
    linewidth=1,
    transform=ax2.transAxes
)
ax2.add_patch(formula_box)
ax2.text(0.5, formula_y + 0.02,
         '$p(t) = \\frac{|\\{i : \\hat{y} + r_i \\leq t\\}|}{n + 1}$',
         ha='center', va='center', fontsize=11,
         color='#333333', transform=ax2.transAxes)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================
output_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(output_dir, 'fig_3_8_cpd_visualization.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(output_dir, 'fig_3_8_cpd_visualization.pdf'),
            bbox_inches='tight', facecolor='white')
print("Saved fig_3_8")
plt.close()