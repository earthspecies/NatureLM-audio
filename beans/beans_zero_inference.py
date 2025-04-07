import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from NatureLM.config import Config
from NatureLM.models import NatureLM
from NatureLM.processors import NatureLMAudioEvalProcessor, NatureLMInferenceDataset, collater
from NatureLM.storage_utils import GSPath, is_gcs_path
from NatureLM.utils import move_to_device

DEFAULT_MAX_LENGTH_SECONDS = 10


def load_beans_cfg(cfg_path: str | Path):
    with open(cfg_path, "r") as cfg_file:
        beans_cfg = json.load(cfg_file)
    return beans_cfg


def main():
    parser = argparse.ArgumentParser("Run BEANS-Zero inference")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--beans_config_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--project_id", type=str, help="Google cloud project id, oka...")
    args = parser.parse_args()

    # load model
    print("Loading model")
    model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
    model = model.to("cuda:0").eval()
    model.llama_tokenizer.pad_token_id = model.llama_tokenizer.eos_token_id

    cfg = Config.from_sources(args.model_config_path)
    # cfg.generate.temperature = None  # do_sample is False

    # Load data
    if is_gcs_path(args.data_path):
        data_path = GSPath(args.data_path)
        ds = load_dataset("arrow", data_files=data_path / "*.arrow", streaming=False,
                          split="train",
                          name="beans-zero",
                          storage_options={"project": args.project_id})
    else:
        data_path = Path(args.data_path)
        ds = load_from_disk(data_path)

    print(f"Loaded dataset with {len(ds)} samples")

    # Load BEANS config
    beans_cfg = load_beans_cfg(args.beans_config_path)
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
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collater,
            num_workers=args.num_workers,
        )

        for batch in tqdm(dl, total=len(dl)):
            batch = move_to_device(batch, "cuda:0")

            output = model.generate(batch, cfg.generate, batch["prompt"])
            outputs["prediction"].extend(output)
            outputs["id"].extend(batch["id"])
            outputs["dataset_name"].extend([dataset_name] * len(batch["id"]))
            outputs["label"].extend(batch["label"])

        # save intermediate results as dataframe
        df = pd.DataFrame(outputs)
        df.to_json(args.output_path, orient="records", lines=True)
        print(f"Saved intermediate results to {args.output_path}")
    #     torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
