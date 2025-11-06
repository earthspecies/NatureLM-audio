from beans_zero.post_processor import EvalPostProcessor
import pandas as pd
import json
from pathlib import Path

folders_list =  [
    "results/closed_set_classification_2",
    "results/closed_set_classification_all",
]

outputs_loaded = dict()
for folder in folders_list:
    print(folder)
    name = folder.split("/")[-1]
    outputs_loaded[name] = []
    for i in range(0, 11):
        scaling_output = []
        with open(Path(folder) / f"beans_zero_eval_unseen-family-cmn_query0_lora{i:02d}0.jsonl", "r") as f:
            for line in f:
                entry = json.loads(line)
                scaling_output.append((entry["prediction"], entry["label"]))
        outputs_loaded[name].append(scaling_output)

out_of_set_counts = dict()

for dataset in outputs_loaded.keys():
    out_of_set_counts[dataset] = []
    for i in range(0, 11):
        outputs, labels = zip(*outputs_loaded[dataset][i])
        outputs = [out.replace("The common name for the focal species in the audio is", "") for out in outputs] # Let's be graceful and remove this prefix if it exists
        results = {"prediction": outputs, "label": labels, "dataset_name": ["unseen-family-cmn"] * len(outputs), "id": list(range(len(outputs)))}
        results_df = pd.DataFrame(results)
            
        labels = results_df["label"].to_list()

        processor = EvalPostProcessor(target_label_set=set(labels), task="detection") # Use detection to get "out of set" handling
        predictions = processor(results_df["prediction"].to_list())

        label_set = set(labels)

        not_in_set = 0
        for pred, label in zip(predictions, labels):
            if pred not in label_set:
                not_in_set += 1

        out_of_set_counts[dataset].append(1 - not_in_set/len(labels))

from matplotlib import pyplot as plt

plt.figure(figsize=(10, 6))

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
# Publication-quality styling
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

marker_kwargs = dict(marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5)
colors = sns.color_palette("Set2", n_colors=len(folders_list))
# Golden ratio proportions for aesthetically pleasing layout

golden_ratio = 1.618
width = 10
height = width / golden_ratio  # ≈ 3.09
fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

i = 0
for dataset, errors in out_of_set_counts.items():
    xs = [x / 10 for x in range(0, 11)]

    if i == 0:
        label = "Closed-Set Classification (2 classes)"
    else:
        label = "Closed-Set Classification (All classes)"

    #ax.bar(xs, errors, label=label, color=colors[i])
    ax.plot(xs, errors, label=label, color=colors[i], **marker_kwargs)
    i += 1

ax.set_ylim(bottom=0, top=None)
ax.set_xlabel('Scaling factor', fontsize=24)
ax.set_ylabel('Class in set percentage', fontsize=24)
ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.25)
ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.15)
ax.tick_params(axis='both', which='both', length=4, width=0.8, labelsize=24)
ax.legend(frameon=False, fontsize=24)
sns.despine(fig, ax,trim=True)
plt.title("Class in Set Percentage vs Scaling Factor for Different Setups", fontsize=28)
plt.savefig(f'correct_classes_prediction_percentage.pdf', bbox_inches='tight')
plt.clf()
