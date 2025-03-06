"""
Calculate retrieval metrics
using the HuggingFace CLAP implementation.

Note: this script may give very slightly different (~0.1%) results from the paper
due to slightly different audio preprocessing. To recreate the exact results, use
CLAP/experiment_scripts/eval_biolingual.sh
"""

# from CLAP.src.laion_clap.training.train import get_metrics_biolingual
import re
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, ClapModel

from beans.beans.datasets import ClassificationDataset
from beans.beans.gpt2 import Gpt2Encoder


def get_metrics_biolingual(
    audio_features,
    text_features,
    logit_scale_a,
    logit_scale_t=None,
    captions=None,
):
    """
    Calculate precision@1 and mean-average precision
    for text-to-audio and audio-to-text rankings.
    Considers answers relevant based on equivalent captions,

    """
    print(f"audio f shape {audio_features.shape} text f shape {text_features.shape} captions len {len(captions)}")
    print("logit scale a", logit_scale_a)
    print("logit scale t", logit_scale_t)
    metrics = {}
    logits_per_audio = (logit_scale_a * audio_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_audio.t().detach().cpu()
    # In get_metrics_biolingual function

    assert (
        len(captions) == audio_features.shape[0] == text_features.shape[0]
    ), "Mismatched dimensions between captions and features"
    assert logits_per_audio.shape == logits_per_text.t().shape, "Mismatched dimensions between logits"

    labels = torch.arange(audio_features.shape[0]).long()
    # Change the loss from two terms into four terms with 2x2 combined CE loss
    total_loss = (F.cross_entropy(logits_per_audio, labels) + F.cross_entropy(logits_per_text, labels)) / 2

    metrics["cumulative_loss"] = total_loss.item()
    metrics["num_samples"] = audio_features.shape[0]

    logits = {"audio_to_text": logits_per_audio, "text_to_audio": logits_per_text}
    print("logits are", logits)

    print("captions len", len(captions))
    duplicates = get_duplicates_matrix(captions)
    assert duplicates.shape == (len(captions), len(captions)), "Invalid shape for duplicates matrix"
    assert torch.all(torch.eq(duplicates, duplicates.t())), "Duplicates matrix should be symmetric"

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        ranks = torch.zeros_like(ranking)
        for i in range(ranking.size(0)):
            ranks[i, ranking[i]] = torch.arange(ranking.size(1))

        relevant_counts_at_k = {k: np.zeros(len(text_features), dtype=float) for k in [1, 3, 5, 10]}
        precision_at_k = {k: np.zeros(len(text_features), dtype=int) for k in [1, 3, 5, 10]}
        ap_at_10 = np.zeros(len(text_features))

        for i in range(len(text_features)):
            equivalent_classes = duplicates[i]
            for k in relevant_counts_at_k:
                assert relevant_counts_at_k[k][i] <= k, "There cannot be more than k relevant items in top k"
                relevant_items_at_k = ranks[i, equivalent_classes] < k
                relevant_counts_at_k[k][i] = relevant_items_at_k.sum().item()
                precision_at_k[k][i] = relevant_counts_at_k[k][i] / k
                assert relevant_counts_at_k[k][i] <= k, "22There cannot be more than k relevant items in top k"

            # Compute AP@10
            temp_precision = 0
            count_relevant_items = 0
            for rank in range(10):  # top
                if equivalent_classes[ranking[i, rank]]:  # Check if the item at the current rank is relevant
                    count_relevant_items += 1
                    temp_precision += count_relevant_items / (rank + 1)  # precision@rank for the current item
            ap_at_10[i] = temp_precision / min(
                10, len(equivalent_classes[equivalent_classes])
            )  # If less than 10 items, adjust denominator

        total_relevant_items = duplicates.sum(axis=1).detach().cpu().numpy()
        print(metrics)
        for k in [1, 3, 5, 10]:
            # Compute Recall@k
            metrics[f"{name}_R@{k}"] = (relevant_counts_at_k[k] / np.maximum(total_relevant_items, 1)).mean()
            # Compute Precision@k
            metrics[f"{name}_P@{k}"] = precision_at_k[k].mean()

        # Compute MAP@10
        metrics[f"{name}_MAP@10"] = ap_at_10.mean()

    print("finished metrics!")
    print("metrics!", metrics)
    return metrics


def clean_caption(caption):
    if caption.startswith("The sound of a "):
        caption = caption[len("The sound of a ") :]
    elif caption.startswith("The sound of an "):
        caption = caption[len("The sound of an ") :]
    return caption.strip()


def get_duplicates_matrix(captions: List[str]):
    captions = [clean_caption(caption) for caption in captions]
    duplicates_matrix = np.array([[cap1 == cap2 for cap2 in captions] for cap1 in captions])
    return torch.from_numpy(duplicates_matrix)


MODEL_IDENTIFIER = "laion/clap-htsat-unfused"
TEST_SET = "test_set.csv"

KEYS_TO_MODIFY_MAPPING = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}


def rename_state_dict(state_dict, exclude_text=False):
    state_dict = {(k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in state_dict.items()}

    model_state_dict = {}

    sequential_layers_pattern = r".*sequential.(\d+).*"
    text_projection_pattern = r".*_projection.(\d+).*"

    for key, value in state_dict.items():
        if exclude_text and "text_branch" in key:
            continue
        # check if any key needs to be modified
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(sequential_layers_pattern, key):
            # replace sequential layers with list
            sequential_layer = re.match(sequential_layers_pattern, key).group(1)

            key = key.replace(f"sequential.{sequential_layer}.", f"layers.{int(sequential_layer)//3}.linear.")
        elif re.match(text_projection_pattern, key):
            projecton_layer = int(re.match(text_projection_pattern, key).group(1))

            # Because in CLAP they use `nn.Sequential`...
            transformers_projection_layer = 1 if projecton_layer == 0 else 2

            key = key.replace(f"_projection.{projecton_layer}.", f"_projection.linear{transformers_projection_layer}.")

        if "audio" and "qkv" in key:
            # split qkv into query key and value
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            model_state_dict[key.replace("qkv", "query")] = query_layer
            model_state_dict[key.replace("qkv", "key")] = key_layer
            model_state_dict[key.replace("qkv", "value")] = value_layer
        else:
            model_state_dict[key] = value

    return model_state_dict


def compute_tta():
    device = "cuda"

    test_df = pd.read_csv(TEST_SET)
    model = ClapModel.from_pretrained(MODEL_IDENTIFIER).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_IDENTIFIER)
    all_captions = test_df["caption"].tolist()
    print("all captions", all_captions[0:9], len(all_captions))
    checkpoint = torch.load("/home/davidrobinson/biolingual-2/CLAP/models/bl1.5.pt", map_location="cpu")
    # checkpoint = torch.load("/home/davidrobinson/biolingual-2/CLAP/logs/2024_02_06-20_27_16-model_HTSAT-tiny-lr_3e-05-b_130-j_7-p_fp32/checkpoints/epoch_latest.pt", map_location="cpu")
    # checkpoint = torch.load("denim-elevator-413.pt", map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = rename_state_dict(state_dict)

    model.load_state_dict(state_dict, strict=False)

    dataset = ClassificationDataset(
        metadata_path=TEST_SET,
        num_labels=len(all_captions),
        labels=all_captions,
        unknown_label="Mississippi Kite",
        sample_rate=48000,
        max_duration=10,
        feature_type="waveform",
    )
    print("made dataset")
    dataloader = DataLoader(
        dataset=dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True
    )
    print("made dataloader")

    gpt_model = AutoModel.from_pretrained("/home/davidrobinson/biolingual-2/CLAP/models/gpt_clap_bl1.5.pt")
    gpt_tokenizer = AutoTokenizer.from_pretrained(
        "/home/davidrobinson/biolingual-2/sapbert/train/models/alignment_gpt_species3"
    )
    # gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt_tokenizer.add_tokens("<|endoftext|>")
    gpt_tokenizer.add_special_tokens({"pad_token": "!"})
    gpt_encoder = Gpt2Encoder(gpt_model, gpt_tokenizer, text_projection=model.text_projection).to(device)

    audio_embeds = []
    text_embeds = []

    for audios, captions in tqdm(dataloader):
        with torch.no_grad():
            x = [s.cpu().numpy() for s in audios]
            print("captions batch", captions)
            inputs = processor(audios=x, text=captions, return_tensors="pt", sampling_rate=48000, padding=True).to(
                device
            )
            model_outputs = model(**inputs)
            audio_embeds.extend(model_outputs.audio_embeds.detach().cpu())
            labels = [text + " <|endoftext|>" for text in captions]
            be = gpt_tokenizer(labels, return_tensors="pt", padding=True)
            text_out = gpt_encoder(be)
            text_embeds.extend(text_out.detach().cpu())

    audio_features = torch.stack(audio_embeds)
    print("audio features shape", audio_features.shape)
    text_features = torch.stack(text_embeds)
    with torch.no_grad():
        get_metrics_biolingual(
            audio_features=audio_features,
            text_features=text_features,
            logit_scale_a=model.logit_scale_a.exp().cpu(),
            captions=all_captions,
        )


if __name__ == "__main__":
    compute_tta()
