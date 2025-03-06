from typing import List

import torch
import torch.nn as nn
import torchvision

from NatureLM.config import Config
from NatureLM.models.NatureLM import NatureLM

from .biolm import BioALLM

LLAMA = True

CUSTOM_ROBERTA = False
GPT = True


class ResNetClassifier(nn.Module):
    def __init__(self, model_type, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        if model_type.startswith("resnet50"):
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.resnet = torchvision.models.resnet50(weights=weights if pretrained else None)
        elif model_type.startswith("resnet152"):
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            self.resnet = torchvision.models.resnet152(weights=weights if pretrained else None)
        elif model_type.startswith("resnet18"):
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.resnet = torchvision.models.resnet18(weights=weights if pretrained else None)
        else:
            assert False

        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)  # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)  # -> (B, 3, F, L)
        x /= x.max()  # normalize to [0, 1]
        # x = self.transform(x)

        x = self.resnet(x)
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits


class VGGishClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False):
        super().__init__()

        self.vggish = torch.hub.load("harritaylor/torchvggish", "vggish")
        self.vggish.postprocess = False
        self.vggish.preprocess = False

        self.linear = nn.Linear(in_features=128, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(x)
        out = out.reshape(batch_size, -1, out.shape[1])
        outs = out.mean(dim=1)
        logits = self.linear(outs)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits


class LLMZeroShotClassifier(nn.Module):
    def __init__(
        self, model_config_path: str, labels: List[str], multi_label=False, match_to_labels=True, loss_ranking=False
    ):
        super().__init__()
        config = Config.from_sources(yaml_file=model_config_path)
        salmon = NatureLM.from_config(config.model).to("cuda:0").eval()
        tokenizer = salmon.llama_tokenizer
        self.allm = BioALLM(
            salmon,
            tokenizer,
            config,
            multi_label=multi_label,
            labels=labels,
            match_to_labels=match_to_labels,
            loss_ranking=loss_ranking,
        )
        self.labels = labels

        self.loss_func = nn.CrossEntropyLoss()
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, x, texts, padding_mask=None, samples=None):
        x = [s.cpu().numpy() for s in x]
        preds = self.allm(x, texts, self.labels, padding_mask=padding_mask, samples=samples)
        logits_per_audio = []
        for pred in preds:
            logits_for_audio = torch.tensor([1 if label == pred else 0 for label in self.labels])
            logits_per_audio.append(logits_for_audio)
        logits_per_audio = torch.stack(logits_per_audio)
        return preds, logits_per_audio
