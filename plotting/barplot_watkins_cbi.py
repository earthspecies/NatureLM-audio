import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
import seaborn as sns

# Publication-quality styling
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

# Bar plot for scientific, common, combined, for both watkins and cbi datasets
labels = ["Common", "Scientific", "Combined"]
datasets = ["Watkins", "CBI"]

# Values: rows are datasets (Watkins, CBI), columns are categories (common, scientific, combined)
values = np.array([
    [0.79, 0.62, 0.10],  # Watkins
    [0.77, 0.70, 0.03]   # CBI
])
colors = sns.color_palette("Set2", n_colors=2)  # Soft, professional - one color per dataset
bar_width = 0.35  # Slightly reduced for more elegant proportions
x = np.arange(len(labels))  # label locations

# Golden ratio proportions for aesthetically pleasing layout
golden_ratio = 1.618
width = 10
height = width / golden_ratio  # ≈ 3.09
fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

bars1 = ax.bar(x - bar_width/2, values[0], width=bar_width, color=colors[0], edgecolor="black", linewidth=0.8, label="Watkins")
bars2 = ax.bar(x + bar_width/2, values[1], width=bar_width, color=colors[1], edgecolor="black", linewidth=0.8, label="CBI")

# Annotate bars with values (decimal style to match other plots)
for bars, dataset_vals in zip([bars1, bars2], values):
    for bar, val in zip(bars, dataset_vals):
        ax.annotate(f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=24,
                    fontweight='regular',
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')])

ax.set_ylabel("Accuracy", fontsize=24, fontweight='regular')
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=24, fontweight='regular')
# ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.25)
ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.15)

# Clean spines and ticks to match style
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis='y', which='both', length=4, width=0.8, labelsize=24)
ax.tick_params(axis='x', which='both', length=4, width=0.8, labelsize=24)

ax.legend(frameon=False, loc='upper right', fontsize=24)

sns.despine(fig, ax,trim=True, offset=10)

# Save to PDF alongside others
fig.savefig('scientific_common_combined_bar.pdf', bbox_inches='tight')