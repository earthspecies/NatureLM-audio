import csv
import json
import os
from dataclasses import dataclass, field

import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from beans.generative_evaluation import evaluate
from beans.models import LLMZeroShotClassifier
from NatureLM.dataset import NatureLMDataset

BASE_ANNOTATION_PATH = "/home/davidrobinson/foundation-model-data/"
BASE_DATA_PATH = "/home/davidrobinson/foundation-model-data/audio"
CLASSIFICATION_PROMPT = "What is the common name for the focal species in the audio?"
CAPTIONING_PROMPT = "<Audio><AudioHere></Audio> Caption the audio, using the common name for any animal species."
DETECTION_PROMPT = "<Audio><AudioHere></Audio> What are the common names for the species in the audio, if any?"

DATASETS_TO_RUN = [
    # BEANS detection
    "hiceas",
    "hainan-gibbons",
    "rfcx",
    "enabirds",
    "dcase",
    # BEANS classification
    "watkins",
    "cbi",
    "cbi-context",
    "cbi-notes",
    "humbugdb",
    "esc50",
    # BirdSet
    "nes_classification_common",
    "nes_classification_sci",
    "nes_classification_tax",
    "per_classification_sci",
    "nbp_classification_sci",
    "pow_classification_sci",
    # BEANS Zero New
    "zf-nbirds",
    "xeno-lifestage",
    "animalspeak-call-type-classification",
    "animalspeak-caption-common",
    # Unseen Species
    "unseen-species-sci-species",
    "unseen-species-taxonomic-species",
    "unseen-species-sci-genus",
    "unseen-species-taxonomic-genus",
    "unseen-species-sci-family",
    "unseen-species-taxonomic-family",
    "unseen-species-sci-class",
    "unseen-species-taxonomic-class",
]

CONFIGURATIONS_TO_RUN = {
    "v1_s2_e96": "/home/davidrobinson/audio-to-text-llm/configs/v1/decode_config.yaml",
}


def load_annotations(annotation_path: str):
    with open(annotation_path) as annotation_handle:
        try:
            output_annotation = json.load(annotation_handle)
        except Exception:
            output_annotation = [json.loads(line) for line in annotation_handle.readlines()]

    return output_annotation


def read_datasets(path):
    with open(path) as f:
        datasets = yaml.safe_load(f)

    return {d["name"]: d for d in datasets}


@dataclass
class TestingConfig:
    model_config_path: str = "/home/davidrobinson/audio-to-text-llm/configs/decode_config.yaml"
    datasets_path: str = "/home/davidrobinson/foundation-model-data/beans_datasets.yml"
    batch_size: int = 16
    num_workers: int = 8
    run_name: str = "base"
    output_base_path: str = field(init=False)  # Path will be set based on run_name
    loss_ranking = False

    def __post_init__(self):
        # Set the output_base_path based on the run_name after initialization
        self.output_base_path = f"/home/davidrobinson/audio-to-text-llm/results/{self.run_name}"
        # Ensure the output directory exists
        os.makedirs(self.output_base_path, exist_ok=True)


def inference(config, labels_for_task, annotation_path: str, task: str, output_path: str, max_length_seconds: int):
    output_annotation = load_annotations(annotation_path)
    is_detection = task == "detection"

    labels_to_match_to = labels_for_task
    if labels_for_task is None:
        assert not is_detection
        labels_for_task = set([row["text"] for row in output_annotation["annotation"]])
        labels_to_match_to = labels_for_task
    if is_detection:
        labels_to_match_to += ["None"]

    model = LLMZeroShotClassifier(
        config.model_config_path,
        labels_to_match_to,
        is_detection,
        match_to_labels=task != "captioning",
        loss_ranking=config.loss_ranking,
    )
    dataset = NatureLMDataset(annotation_path, cropping="start", max_length_seconds=max_length_seconds)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=dataset.collater,
        drop_last=False,
    )

    all_predictions = []
    all_true_labels = []

    for batch_index, batch in tqdm(enumerate(loader), total=len(loader)):
        audios, prompts, padding_mask, labels, indices = (
            batch["raw_wav"],
            batch["prompt"],
            batch["padding_mask"],
            batch["text"],
            batch["index"],
        )

        if (
            is_detection and len(labels_to_match_to) > 8
        ):  # Current workaround option: for prompts with many options, don't use options.
            prompts = [DETECTION_PROMPT for t in prompts]
            batch["prompt"] = prompts

        # try:
        predictions, _ = model(audios, prompts, padding_mask=padding_mask, samples=batch)
        # predictions, _ = model(samples)
        # except Exception:
        # print()
        # predictions = ["" for t in prompts]  # TODO: Corner case (humbugdb)

        for index, prediction, label in zip(indices, predictions, labels):
            print(f"pred is {prediction} label is {label}")
            true_label = output_annotation["annotation"][index]["text"]
            assert label == true_label

            all_true_labels.append(label)

            # Check if the model returned a dictionary (label-to-score) or a text prediction
            if isinstance(prediction, dict):
                # If it's a dictionary, assume it's a label-to-score mapping
                output_annotation["annotation"][index]["prediction"] = prediction
            elif isinstance(prediction, str):
                # If it's a string, assume it's a text-based prediction
                output_annotation["annotation"][index]["prediction"] = prediction
            else:
                # Handle corner case
                output_annotation["annotation"][index]["prediction"] = ""

            all_predictions.append(prediction)

    metrics = evaluate(all_predictions, all_true_labels, task=task, labels=labels_for_task)
    output_annotation["metrics"] = metrics

    with open(output_path, "w") as output_handle:
        json.dump(output_annotation, output_handle, indent=2)
        print("saving predictions to", output_path)  # SAVE preds for separate eval

    return all_predictions, all_true_labels, metrics


def run_inference(config, sr_48k=False):
    datasets = read_datasets(config.datasets_path)
    metrics_data = []  # List to store dataset names and metrics

    for dataset_name, dataset in datasets.items():
        if "llm_data" not in dataset or dataset_name not in DATASETS_TO_RUN:
            continue
        # print(f"Running inference on {dataset_name}")
        if "human_labels" in dataset or "labels" in dataset:
            labels = dataset["human_labels"] if "human_labels" in dataset else dataset["labels"]
        else:
            labels = None
        # print("labels for dataset", dataset_name, labels)
        task = dataset["type"]
        if sr_48k:
            annotation_path = dataset["llm_data"].replace(".jsonl", "_processed_48k.jsonl")
        else:
            annotation_path = dataset["llm_data"].replace(".jsonl", "_processed.jsonl")
        annotation_path = os.path.join(BASE_ANNOTATION_PATH, annotation_path)
        output_path = annotation_path.split(".jsonl")[0] + "_predictions.jsonl"
        max_length_seconds = dataset["max_duration"] if "max_duration" in dataset else 10

        predictions, true_labels, metrics = inference(
            config,
            labels,
            annotation_path=annotation_path,
            task=task,
            output_path=output_path,
            max_length_seconds=max_length_seconds,
        )

        # Append dataset name and metrics to the list for CSV export
        metrics_data.append({"dataset_name": dataset_name, "metrics": metrics})

        # Save the metrics data to a CSV file
        csv_output_path = os.path.join(config.output_base_path, f"{config.run_name}_metrics.csv")
        with open(csv_output_path, mode="w", newline="") as csvfile:
            fieldnames = ["dataset_name", "metrics"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for data in metrics_data:
                writer.writerow(data)

        print(f"Metrics saved to {csv_output_path}")


if __name__ == "__main__":
    for name, model_config_path in CONFIGURATIONS_TO_RUN.items():
        config = TestingConfig(run_name=name, model_config_path=model_config_path)
        os.makedirs(config.output_base_path, exist_ok=True)
        run_inference(config)
