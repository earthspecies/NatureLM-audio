from beans_zero.evaluate import compute_metrics
import json
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
from utils import *

folders_list =  [
    "results/closed_set_classification_2",
    "results/closed_set_classification_all",
    "results/in_context_learning_1_example",
    "results/in_context_learning_5_examples",
    # "results/textual_in_context_learning",
]

baselines = {}
outputs_loaded = dict()
for folder in folders_list:
    print(folder)
    name = folder.split("/")[-1]
    outputs_loaded[name] = []
    for i in range(0, 11):
        scaling_output = []
        with open(Path(folder) /  f"beans_zero_eval_unseen-family-cmn_query0_lora{i:02d}0.jsonl", "r") as f:
            for line in f:
                entry = json.loads(line)
                scaling_output.append((entry["prediction"], entry["label"]))
        outputs_loaded[name].append(scaling_output)
        if i == 0:
            # Compute baselines
            labels = [label for _, label in scaling_output]
            class_counts = pd.Series(labels).value_counts().sort_index().tolist()
            baselines[name] = (macro_f1_baselines(class_counts), accuracy_baseline(class_counts))

metrics_per_setup = dict()

for dataset in outputs_loaded.keys():
    metrics_per_setup[dataset] = []
    for i in range(0, 11):
        outputs, labels = zip(*outputs_loaded[dataset][i])
        outputs = [out.replace("The common name for the focal species in the audio is", "") for out in outputs] # Let's be graceful and remove this prefix if it exists
        results = {"prediction": outputs, "label": labels, "dataset_name": ["unseen-family-cmn"] * len(outputs), "id": list(range(len(outputs)))}
        results_df = pd.DataFrame(results)
            
        metrics = compute_metrics(results_df, verbose=False)
        metrics_per_setup[dataset].append(metrics["unseen-family-cmn"])
        print(f"Dataset: {dataset}, Scaling: {i * 10}%")
        print(metrics["unseen-family-cmn"]["Accuracy"], metrics["unseen-family-cmn"]["F1 Score"])

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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

tests = list(metrics_per_setup.keys())

for files, names, postfix in [
                     (["closed_set_classification_2"], ["Closed Set Classification"], "closed_set_2"),
                     (["closed_set_classification_all"], ["Closed Set Classification"], "closed_set_all"),
                     (["closed_set_classification_2", "in_context_learning_1_example", "in_context_learning_5_examples"], ["k=0", "k=1", "k=5"], "in_context_audio"),
                    #  (["textual_in_context_learning"], ["Merged model"], "in_context_text")
                     ]:
    for ax_index, plot_type in enumerate(["F1 Score", "Accuracy"]):
        golden_ratio = 1.618
        width = 10
        height = width / golden_ratio  # ≈ 3.09
        # width *= 2  # For two subplots
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height), constrained_layout=True)
        i = 0
        for df, filename in zip(metrics_per_setup.values(), metrics_per_setup.keys()):
            if not filename in files:
                continue
            # if filename.endswith("after_tag.out"):
            #     continue
            xs = [x / 10 for x in range(0, 11)]
            accuracies = [metric[plot_type] for metric in df]
            
            line_style = '-'

            filename = names[i]
            
            ax.plot(xs, accuracies, label=filename, color=colors[i], linestyle=line_style, **marker_kwargs)
            for xv, yv in zip(xs, accuracies):
                ax.annotate(
                    f"{yv:.2f}", (xv, yv), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=16, color='black',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
                )
            i += 1


        # Add line for majority class
        macros = baselines[files[0]][ax_index]
        ax.axhline(y=macros[0], color="grey", linestyle='--', label=f'Majority class baseline', linewidth=1.5)

        ax.axhline(y=macros[1], color="grey", linestyle='-.', label=f'Random baseline', linewidth=1.5)

        max_val = max([metric[plot_type] for filename,df in metrics_per_setup.items() for metric in df if filename in files])
        max_val = (min(int(max_val * 10) + 2, 11)) / 10.0
        ax.set_yticks(np.arange(0, max_val, 0.1))
        print(max_val)

        ax.set_ylim(bottom=0, top=None)
        ax.set_xlabel('Scaling factor', fontsize=24)
        ax.set_ylabel(plot_type, fontsize=24)
        ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.25)
        ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.15)
        ax.tick_params(axis='both', which='both', length=4, width=0.8, labelsize=24)
        ax.legend(ncol=2, frameon=False, fontsize=22)
        sns.despine(fig, ax,trim=True)
        plt.savefig(f'{plot_type.replace(" ", "_").lower().replace("_score", "")}_unseen_cmn_family_{postfix}.pdf', bbox_inches='tight')
        plt.clf()