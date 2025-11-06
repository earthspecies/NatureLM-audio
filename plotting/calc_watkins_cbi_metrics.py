from beans_zero.evaluate import compute_metrics
import json
import pandas as pd
import seaborn as sns

# Load mappings
watkins_mapping = dict()
with open("watkins_vernacular_scientific_mapping.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        watkins_mapping[entry["label"]] = entry["scientific_name"]

cbi_mapping = dict()
with open("cbi_vernacular_scientific_mapping.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        cbi_mapping[entry["label"]] = entry["scientific_name"]

print("Loaded mappings for Watkins and CBI.")
print(list(watkins_mapping.items())[:10])
print(list(cbi_mapping.items())[:10])

datasets = ["watkins", "cbi"]
types = ["common", "scientific", "combined"] # 0 = common, 1 = scientific, 2 = combined

outputs_loaded = dict()
for dataset in datasets:
    dataset_outputs = dict()
    outputs_loaded[dataset] = dataset_outputs
    for type_ in types:
        path = f"results/{dataset}/beans_zero_eval_"
        if dataset == "watkins":
            path += "watkins_"
        elif dataset == "cbi":
            path += "cbi_"
        if type_ == "combined":
            path += "query2_"
        elif type_ == "scientific":
            path += "query1_"
        else:
            path += "query0_"
        type_outputs = []
        dataset_outputs[type_] = type_outputs
        for i in range(0, 11):
            scaling_output = []
            type_outputs.append(scaling_output)
            with open(path + f"lora{i:02d}0.jsonl", "r") as f:
                for line in f:
                    entry = json.loads(line)
                    scaling_output.append((entry["prediction"], entry["label"]))

print(outputs_loaded["watkins"]["common"][0][0:5])

metrics_per_setup = dict()

for dataset in datasets:
    metrics_per_setup[dataset] = dict()
    for type_ in types:
        metrics_per_setup[dataset][type_] = []
        for i in range(0, 11):
            outputs, labels = zip(*outputs_loaded[dataset][type_][i])
            outputs = [out.replace("The common name for the focal species in the audio is", "") for out in outputs]
            outputs = [out.replace("The scientific name for the focal species in the audio is", "") for out in outputs]
            outputs = [out.split(",")[0] for out in outputs]
            if type_ == "scientific":
                mapping = watkins_mapping if dataset == "watkins" else cbi_mapping
                labels = [mapping[label] for label in labels]
            elif type_ == "combined":
                mapping = watkins_mapping if dataset == "watkins" else cbi_mapping
                labels = [mapping[label] + " : " + label for label in labels]

            results = {"prediction": outputs, "label": labels, "dataset_name": [dataset] * len(outputs), "id": list(range(len(outputs)))}
            results_df = pd.DataFrame(results)
                
            metrics = compute_metrics(results_df, verbose=False)
            metrics_per_setup[dataset][type_].append(metrics[dataset])
            print(f"Dataset: {dataset}, Type: {type_}, Scaling: {i * 10}%")
            print(metrics[dataset]["Accuracy"], metrics[dataset]["F1 Score"])

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from tueplots import bundles
# Publication-quality styling
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

marker_kwargs = dict(marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5)

colors = sns.color_palette("Set2", n_colors=3)  # Soft, professional

# Golden ratio proportions for aesthetically pleasing layout
golden_ratio = 1.618
width = 10
height = width / golden_ratio  # ≈ 3.09
fig, axs = plt.subplots(1, 2, figsize=(width * 2, height), constrained_layout=True)

# # Color per dataset
# colors = {
#     "watkins": "tab:blue",
#     "cbi": "tab:orange"
# }
# # Line styles per type
# line_styles = {
#     "common": "-",
#     "scientific": "--",
#     "combined": ":"
# 

for dataset_name, ax in zip(datasets, axs):
    for i, type_ in enumerate(types):
        xs = [j / 10 for j in range(0, 11)]
        accuracies = []
        for metrics in metrics_per_setup[dataset_name][type_]:
            accuracies.append(metrics["Accuracy"])

        # line_kwargs = dict(color=colors[dataset_name], linestyle=line_styles[type_], linewidth=1.5)

        ax.plot(xs, accuracies, label=type_.capitalize(), color=colors[i], **marker_kwargs)#, **line_kwargs)
        for xv, yv in zip(xs, accuracies):
            ax.annotate(
                f"{yv:.2f}", (xv, yv), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=16, color='black',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
            )
    
    # ax.set_ylim(0, 0.85)
    ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.25)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.15)
    ax.tick_params(axis='both', which='both', length=4, width=0.8, labelsize=24)
    ax.set_title("Watkins" if dataset_name=="watkins" else "CBI", fontsize=24)
    ax.set_xlabel('Scaling factor', fontsize=24)
    ax.grid(True)
    ax.set_ylabel('Accuracy', fontsize=24)
    if dataset_name == "watkins":
        ax.legend(frameon=False, fontsize=24)
    sns.despine(fig, ax,trim=True)
plt.savefig('accuracy_scaling_factors_watkins_cbi.pdf', bbox_inches='tight')

# Generate combined accuracy vs downstream task plot
marker_kwargs = dict(marker='o', markersize=8, markerfacecolor='white', markeredgewidth=3)

colors = sns.color_palette("Set2", n_colors=3)  # Soft, professional

# Golden ratio proportions for aesthetically pleasing layout
golden_ratio = 1.618
width = 10
height = width  # ≈ 3.09
fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

for i, dataset_name in enumerate(datasets):
    common_accuracies = [m["Accuracy"] for m in metrics_per_setup[dataset_name]["common"]]
    scientific_accuracies = [m["Accuracy"] for m in metrics_per_setup[dataset_name]["scientific"]]
    ys = [m["Accuracy"] for m in metrics_per_setup[dataset_name]["combined"]]
    xs = [.5 * (c + s) for c,s in zip(common_accuracies, scientific_accuracies)]

    ax.plot(xs, ys, label="Watkins" if dataset_name=="watkins" else "CBI", color=colors[i], linewidth=2.0, **marker_kwargs)#, **line_kwargs)
    for xv, yv, scale in zip(xs, ys, range(0, 11)):
        txt = f"a=1"
        if scale != 10:
            txt = f"a=.{scale}"

        ax.annotate(
            txt, (xv, yv), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=24*1.5, color=colors[i],
            path_effects=[pe.withStroke(linewidth=4, foreground='white')]
        )
    

ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.25)
ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.15)
ax.tick_params(axis='both', which='both', length=4, width=0.8, labelsize=24*2)
ax.set_xlabel('Training tasks accuracy', fontsize=28*2)
ax.set_ylabel('Combined task accuracy', fontsize=28*2)
ax.legend(frameon=False, fontsize=24*1.5, loc='upper left')
sns.despine(fig, ax,trim=True)
plt.savefig('combined_vs_original_tasks.pdf', bbox_inches='tight')

