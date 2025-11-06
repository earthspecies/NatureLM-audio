from beans_zero.evaluate import compute_metrics
import json
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
from utils import *

dataset = "zf-indiv"
types = ["Original", "Reversed", "No Classes"] # 0 = common, 1 = scientific, 2 = combined
outputs_loaded = dict()
for type_ in types:
    path = f"results/{dataset}/beans_zero_eval_zf-indiv_"
    if type_ == "No Classes":
        path += "query2_"
    elif type_ == "Reversed":
        path += "query1_"
    else:
        path += "query0_"
    type_outputs = []
    outputs_loaded[type_] = type_outputs
    for i in range(10, 11):
        scaling_output = []
        type_outputs.append(scaling_output)
        with open(path + f"lora{i:02d}0.jsonl", "r") as f:
            for line in f:
                entry = json.loads(line)
                scaling_output.append((entry["prediction"], entry["label"]))
# Compute baselines
labels = [label for _, label in scaling_output]
class_counts = pd.Series(labels).value_counts().sort_index().tolist()
acc_baseline_value = accuracy_baseline(class_counts)

metrics_per_setup = dict()
for dataset in outputs_loaded.keys():
    print(dataset)
    outputs, labels = zip(*outputs_loaded[dataset][0])
    outputs = [out if not out.isdigit() else ("One" if int(out) < 2 else "More") for out in outputs] # Let's be graceful and remove this prefix if it exists
    results = {"prediction": outputs, "label": labels, "dataset_name": ["zf-indiv"] * len(outputs), "id": list(range(len(outputs)))}
    results_df = pd.DataFrame(results)
        
    metrics = compute_metrics(results_df, verbose=False)
    metrics_per_setup[dataset] = metrics["zf-indiv"]["Accuracy"]
    print(f"Dataset: {dataset}, Scaling: {i * 10}%")
    print(metrics["zf-indiv"]["Accuracy"], metrics["zf-indiv"]["F1 Score"])

labels = list(metrics_per_setup.keys())
values = list(metrics_per_setup.values())

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from tueplots import bundles
# Publication-quality styling
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

colors = sns.color_palette("Set2", n_colors=3)

# colors = ["tab:blue", "tab:orange", "tab:green"]
bar_width = 0.55
x = np.arange(len(labels))  # label locations

golden_ratio = 1.618
width = 10
height = width / golden_ratio  # ≈ 3.09

fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

bars = ax.bar(x, values, width=bar_width, color=colors, edgecolor="black", linewidth=0.8)

# Annotate bars with values (decimal style to match other plots)
for bar, val in zip(bars, values):
    ax.annotate(f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=24,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

ax.set_ylabel("Classification Accuracy", fontsize=24)
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=24)
ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.25)
ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.15)

# Clean spines and ticks to match style
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis='both', which='both', length=4, width=0.8, labelsize=24)
# Add random baseline as a dashed horizontal line
macros = acc_baseline_value
baseline_line = ax.axhline(y=macros[1], color="grey", linestyle='--', label=f'Random baseline', linewidth=1.5)
ax.legend(handles=[baseline_line], frameon=False, loc='upper right', fontsize=24)

sns.despine(fig, ax, trim=True, offset=10)

# Save to PDF alongside others
fig.savefig('zf_indiv_plot.pdf', bbox_inches='tight')

plt.show()