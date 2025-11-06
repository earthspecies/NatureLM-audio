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

# Bar plot comparing NatureLM-Audio and NatureLM-Audio-Merged
labels = [r"NatureLM-Audio-Merged ($\alpha=0.4$)", "NatureLM-Audio"]
values = [0.28, 0.09]

palette = sns.color_palette("Set2", n_colors=10)
# Create a lighter version of palette[2] for NatureLM-Audio
base_color = palette[2]
lighter_color = tuple(c + (1 - c) * 0.6 for c in base_color[:3]) + (base_color[3],) if len(base_color) == 4 else tuple(c + (1 - c) * 0.6 for c in base_color)
colors = [base_color, lighter_color]  # NatureLM-Audio-Merged (darker), NatureLM-Audio (lighter)
x = np.arange(len(labels))  # label locations

# Golden ratio proportions for aesthetically pleasing layout
golden_ratio = 1.618
width = 10
height = width / golden_ratio  # ≈ 3.09
fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.8)

# Add random baseline
baseline = ax.axhline(y=0.01, color='grey', linestyle='-.', linewidth=1.5, label='Random baseline')

# Annotate bars with values (decimal style to match other plots)
for bar, val in zip(bars, values):
    ax.annotate(f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=24,
                fontweight='regular',
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

ax.set_ylabel("F1 Score", fontsize=24, fontweight='regular')
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

# Add legend for the baseline
ax.legend(frameon=False, fontsize=22)

sns.despine(fig, ax, trim=True, offset=10)

# Save to PDF alongside others
fig.savefig('unseen_cmn_family_all_classes_bar_plot.pdf', bbox_inches='tight')

plt.show()