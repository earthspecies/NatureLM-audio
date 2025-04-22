import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from beans_zero.evaluate import compute_metrics
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from NatureLM.config import Config
from NatureLM.models import NatureLM
from NatureLM.processors import NatureLMAudioEvalProcessor, NatureLMInferenceDataset, collater
from NatureLM.storage_utils import GSPath, is_gcs_path
from NatureLM.utils import move_to_device

DEFAULT_MAX_LENGTH_SECONDS = 10
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
__this_dir = Path(__file__).parent
BEANS_ZERO_CONFIG_PATH = __this_dir.parent / "beans_zero_config.json"
DEFAULT_CONFIG_PATH = __this_dir.parent / "configs" / "inference.yml"


def load_beans_cfg(cfg_path: str | Path):
    with open(cfg_path, "r") as cfg_file:
        beans_cfg = json.load(cfg_file)
    return beans_cfg


def parse_args():
    parser = argparse.ArgumentParser("Run BEANS-Zero inference")
    parser.add_argument(
        "--cfg-path", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the NatureLM model config file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="EarthSpeciesProject/BEANS-Zero",
        help="Path to the BEANS-Zero dataset. If a local or a google cloud bucket path is not provided, will download from the hub.",
    )
    parser.add_argument(
        "--output_path", type=str, default="beans_zero_eval_output.jsonl", help="Path to save the output results"
    )
    parser.add_argument(
        "--beans_zero_config_path", type=str, default=BEANS_ZERO_CONFIG_PATH, help="Path to the BEANS config file"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    args = parser.parse_args()

    return args


def main(
    cfg_path: str | Path,
    beans_zero_config_path: str | Path,
    data_path: str | Path,
    output_path: str | Path,
    batch_size: int,
    num_workers: int = 0,
) -> None:
    """
    Main function to run inference on the BEANS-Zero dataset.

    Args:
        cfg_path (str | Path): Path to the configuration file.
        beans_zero_config_path (str | Path): Path to the BEANS config json file.
        data_path (str | Path): Path to the dataset.
        output_path (str | Path): Path to save the output results.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for DataLoader.
            Default is 0. Currently, its best to stick to 0 because of issues with resampy.
    """
    # load model
    print("Loading model")
    model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
    model = model.to(DEVICE).eval()
    model.llama_tokenizer.pad_token_id = model.llama_tokenizer.eos_token_id
    model.llama_model.generation_config.pad_token_id = model.llama_tokenizer.pad_token_id

    cfg = Config.from_sources(cfg_path)

    # Load data
    if is_gcs_path(data_path):
        data_path = GSPath(data_path)
        ds = load_dataset(
            "arrow",
            data_files=data_path / "*.arrow",
            streaming=False,
            split="train",
            name="beans-zero",
        )
    else:
        data_path = Path(data_path)

        # check if path is local and exists
        if not data_path.exists():
            # load from hub
            ds = load_dataset("EarthSpeciesProject/BEANS-Zero", split="test")
        else:
            ds = load_from_disk(data_path)

    print(f"Loaded dataset with {len(ds)} samples")

    # Load BEANS config
    beans_cfg = load_beans_cfg(beans_zero_config_path)
    print("Loaded BEANS config")

    # extract dataset configs
    components = beans_cfg["metadata"]["components"]
    ds_names = [d["name"] for d in components]
    ds_tasks = [d["task"] for d in components]
    ds_labels = [d["labels"] for d in components]
    ds_max_length_seconds = [d["max_duration"] for d in components]

    outputs = {"prediction": [], "label": [], "id": [], "dataset_name": []}

    for i, dataset_name in enumerate(ds_names):
        subset = ds.select(np.where(np.array(ds["dataset_name"]) == dataset_name)[0])
        print(f"\n======Running inference on {dataset_name} with {len(subset)} samples======")
        print(f"Task: {ds_tasks[i]}")
        if ds_labels[i] is not None:
            print(f"Num labels: {len(ds_labels[i])}")

        max_length_seconds = ds_max_length_seconds[i]
        if max_length_seconds is None:
            max_length_seconds = DEFAULT_MAX_LENGTH_SECONDS
        print(f"Max duration: {max_length_seconds}")

        processor = NatureLMAudioEvalProcessor(
            max_length_seconds=max_length_seconds,
            dataset_name=dataset_name,
            task=ds_tasks[i],
            true_labels=ds_labels[i] or [],
        )

        dl = DataLoader(
            NatureLMInferenceDataset(subset, processor),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collater,
            num_workers=num_workers,
        )

        for batch in tqdm(dl, total=len(dl)):
            batch = move_to_device(batch, DEVICE)

            output = model.generate(batch, cfg.generate, batch["prompt"])
            outputs["prediction"].extend(output)
            outputs["id"].extend(batch["id"])
            outputs["dataset_name"].extend([dataset_name] * len(batch["id"]))
            outputs["label"].extend(batch["label"])

        # save intermediate results as dataframe
        df = pd.DataFrame(outputs)
        df.to_json(str(output_path), orient="records", lines=True)
        print(f"Saved intermediate results to {output_path}")

    # run evaluation
    print("Running evaluation")
    all_metrics = compute_metrics(df, verbose=True)

    # save final results, replace '.jsonl' with _metrics.json
    metrics_path = str(output_path).replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"Saved evaluation results to {metrics_path}")
    print("Evaluation complete!")


if __name__ == "__main__":
    args = parse_args()
    main(
        cfg_path=args.cfg_path,
        beans_config_path=args.beans_zero_config_path,
        data_path=args.data_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
