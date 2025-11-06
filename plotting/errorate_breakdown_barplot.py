from beans_zero.post_processor import EvalPostProcessor
import pandas as pd
import json
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent
output_dir = Path(__file__).parent

folders_list =  [
    str(project_root / "results" / "closed_set_classification_2"),
    str(project_root / "results" / "closed_set_classification_alls"),
]

outputs_loaded = dict()
for folder in folders_list:
    name = Path(folder).name
    outputs_loaded[name] = []
    for i in range(0, 11):
        scaling_output = []
        file_path = Path(folder) / f"beans_zero_eval_unseen-family-cmn_query0_lora{i:02d}0.jsonl"
        with open(file_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                scaling_output.append((entry["prediction"], entry["label"]))
        outputs_loaded[name].append(scaling_output)

error_breakdown = dict()

for dataset in outputs_loaded.keys():
    error_breakdown[dataset] = {"out_of_set": [], "in_set_wrong": []}
    for i in range(0, 11):
        outputs, labels = zip(*outputs_loaded[dataset][i])
        outputs = [out.replace("The common name for the focal species in the audio is", "") for out in outputs] # Let's be graceful and remove this prefix if it exists
        results = {"prediction": outputs, "label": labels, "dataset_name": ["unseen-family-cmn"] * len(outputs), "id": list(range(len(outputs)))}
        results_df = pd.DataFrame(results)
            
        labels = results_df["label"].to_list()

        processor = EvalPostProcessor(target_label_set=set(labels), task="detection") # Use detection to get "out of set" handling
        predictions = processor(results_df["prediction"].to_list())

        label_set = set(labels)

        out_of_set_count = 0
        in_set_wrong_count = 0
        
        for pred, label in zip(predictions, labels):
            if pred not in label_set:
                out_of_set_count += 1
            elif pred != label:
                in_set_wrong_count += 1

        # Convert to error rates (as percentages)
        total = len(labels)
        error_breakdown[dataset]["out_of_set"].append(out_of_set_count / total)
        error_breakdown[dataset]["in_set_wrong"].append(in_set_wrong_count / total)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Publication-quality styling
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

# Map dataset names to more readable titles and filenames
dataset_info = {
    "closed_set_classification_2": {"title": "2 Classes", "filename": "error_rate_breakdown_2_classes.pdf"},
    "closed_set_classification_all": {"title": "All Classes", "filename": "error_rate_breakdown_all_classes.pdf"}
}

# Generate a plot for each dataset
for dataset_name in error_breakdown.keys():
    # Golden ratio proportions for aesthetically pleasing layout
    golden_ratio = 1.618
    width = 10
    height = width / golden_ratio  # ≈ 3.09
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    
    # Get the data
    out_of_set_errors = error_breakdown[dataset_name]["out_of_set"]
    in_set_wrong_errors = error_breakdown[dataset_name]["in_set_wrong"]
    
    # X positions for bars
    xs = np.arange(11)
    scaling_factors = [x / 10 for x in range(0, 11)]
    
    # Bar width
    bar_width = 0.8
    
    # Colors for the two error types
    color_out_of_set = sns.color_palette("Set2")[1]  # Orange/salmon
    color_in_set_wrong = sns.color_palette("Set2")[5]  # Green
    
    # Create stacked bars
    ax.bar(xs, out_of_set_errors, bar_width, label='"Out-of-set" predictions', color=color_out_of_set)
    ax.bar(xs, in_set_wrong_errors, bar_width, bottom=out_of_set_errors, 
           label='"In-set but wrong" predictions', color=color_in_set_wrong)
    
    ax.set_ylim(bottom=0, top=None)
    ax.set_xlabel('Scaling factor', fontsize=24)
    ax.set_ylabel('Error rate ↓', fontsize=24)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x:.1f}' for x in scaling_factors])
    ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.25, axis='y')
    ax.tick_params(axis='both', which='both', length=4, width=0.8, labelsize=24)
    ax.legend(frameon=False, fontsize=24, loc='upper right')
    sns.despine(fig, ax, trim=True)
    
    # Get title and filename for this dataset
    info = dataset_info[dataset_name]
    
    plt.savefig(output_dir / info["filename"], bbox_inches='tight')
    print(f"Saved plot for {info['title']} to {info['filename']}")
    plt.close(fig)
