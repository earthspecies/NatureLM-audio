import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import wav2vec2_model


class AvesEmbedding(nn.Module):
    def __init__(self, sr, large=False):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        if large:
            config = self.load_config("configs/birdaves_bioxlarge.config")
        else:
            config = self.load_config("configs/birdaves_bioxbase.config")
        self.model = wav2vec2_model(**config, aux_num_out=None)
        state_dict = torch.hub.load_state_dict_from_url(
            "https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.torchaudio.pt",
            map_location=device,
        )
        self.model.load_state_dict(state_dict)
        self.model.feature_extractor.requires_grad_(True)

        # bundle = torchaudio.pipelines.WAV2VEC2_BASE
        # self.model = bundle.get_model()

        self.sr = sr

    def load_config(self, config_path):
        with open(config_path, "r") as ff:
            obj = json.load(ff)

        return obj

    def forward(self, sig, padding_mask):
        # extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
        # print("sig", sig)

        out = self.model.extract_features(sig.float())[0][-1]
        atts = ~padding_mask
        atts = atts.unsqueeze(1).float()
        atts = F.max_pool1d(atts, kernel_size=320, stride=320)
        atts = atts > 0
        padding_mask = ~atts

        return out, padding_mask

    def freeze(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.model.feature_extractor.requires_grad_(False)

    def unfreeze(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        self.model.feature_extractor.requires_grad_(True)
