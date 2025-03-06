# Copyright (2024) Earth Species Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from NatureLM.config import Config
from NatureLM.dataset import NatureLMDataset
from NatureLM.dist_utils import get_rank, init_distributed_mode
from NatureLM.logger import setup_logger
from NatureLM.models import load_model
from NatureLM.runner import Runner
from NatureLM.utils import now_as_str


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main(cfg_path: str | Path, options: list[str] = []):
    # done before init_distributed_mode() to ensure the same job_id is shared across all
    # ranks
    job_id = now_as_str()

    cfg = Config.from_sources(yaml_file=cfg_path, cli_args=options)
    init_distributed_mode(cfg.run)
    cfg.pretty_print()

    setup_seeds(cfg.run)
    setup_logger()

    model = load_model(cfg.model)

    datasets = {
        "train": NatureLMDataset(
            cfg.datasets.train_ann_path,
            max_length_seconds=cfg.datasets.audio_max_length_seconds,
            seed=cfg.run.seed,
            noise_prob=cfg.run.augmentations.noise_prob,
            noise_dirs=cfg.run.augmentations.noise_dirs,
            low_snr=cfg.run.augmentations.low_snr,
            high_snr=cfg.run.augmentations.high_snr,
            time_scale_prob=cfg.run.augmentations.time_scale_prob,
            time_scale=cfg.run.augmentations.time_scale,
            mixup_prob=cfg.run.augmentations.mixup_prob,
            mixup_count=cfg.run.augmentations.mixup_count,
            use_augmentation=cfg.run.augmentations.use_augmentation,
            mask_audio_prob=cfg.run.augmentations.mask_audio_prob,
        ),
        "valid": NatureLMDataset(
            cfg.datasets.valid_ann_path, max_length_seconds=cfg.datasets.audio_max_length_seconds, cropping="start"
        ),
        "test": NatureLMDataset(
            cfg.datasets.test_ann_path, max_length_seconds=cfg.datasets.audio_max_length_seconds, cropping="start"
        ),
    }

    runner = Runner(cfg, model, datasets, job_id)
    runner.train()
