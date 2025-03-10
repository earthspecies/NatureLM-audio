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

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator
from pydantic.v1.utils import deep_update
from pydantic_settings import BaseSettings, CliSettingsSource, YamlConfigSettingsSource

from NatureLM.storage_utils import GSPath, is_gcs_path


class OptimizerConfig(BaseModel, extra="forbid", validate_assignment=True):
    max_epoch: int
    warmup_steps: int
    warmup_start_lr: float = -1
    init_lr: float
    min_lr: float
    weight_decay: float
    beta2: float = 0.999
    max_grad_norm: float | None = None
    max_grad_value: float | None = None
    device: str = "cuda"


class AugmentationsConfig(BaseModel, extra="forbid", validate_assignment=True):
    use_augmentation: bool = False

    noise_prob: float = 0
    noise_dirs: list[Path] | None = None
    low_snr: float = -5
    high_snr: float = 20
    time_scale_prob: float = 0
    time_scale: float = 1.2
    mixup_prob: float = 0
    mixup_count: int = 3
    mask_audio_prob: float = 0


class RunConfig(BaseModel, extra="forbid", validate_assignment=True):
    wandb_enabled: bool = True
    amp: bool = False
    seed: int
    output_dir: Path
    evaluate: bool
    log_freq: int
    epoch_based: bool
    iters_per_epoch: int
    accum_grad_iters: int
    batch_size_train: int
    batch_size_eval: int
    num_workers: int
    custom_metrics: bool
    decode_ratio: float

    device: Literal["cuda", "cpu"] = "cuda"
    use_distributed: bool = False

    world_size: int = 1
    rank: int = 0
    gpu: int | None = None
    dist_backend: Literal["nccl"] = "nccl"
    dist_url: str = "env://"

    optims: OptimizerConfig
    augmentations: AugmentationsConfig


class DatasetsConfig(BaseModel, extra="forbid", validate_assignment=True):
    train_ann_path: Path
    valid_ann_path: Path
    test_ann_path: Path
    audio_max_length_seconds: int

    @field_validator("train_ann_path", "valid_ann_path", "test_ann_path", mode="after")
    @classmethod
    def check_files(cls, path: Path) -> Path:
        if not path.exists():
            raise ValueError(f"File {path} does not exist")
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"File {path} must be a JSONL file")
        return path


class BeatsConfig(BaseModel, extra="forbid", validate_assignment=True):
    input_patch_size: int = -1
    embed_dim: int = 512
    conv_bias: bool = False

    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12
    activation_fn: str = "gelu"

    layer_wise_gradient_decay_ratio: float = 0.6
    layer_norm_first: bool = False
    deep_norm: bool = True

    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.05
    dropout_input: float = 0.0

    conv_pos: int = 128
    conv_pos_groups: int = 16

    relative_position_embedding: bool = True
    num_buckets: int = 320
    max_distance: int = 800
    gru_rel_pos: bool = True

    finetuned_model: bool = True
    predictor_dropout: float = 0.0
    predictor_class: int = 527


class GenerateConfig(BaseModel, extra="forbid", validate_assignment=True):
    max_new_tokens: int
    num_beams: int
    do_sample: bool
    min_length: int
    temperature: float
    repetition_penalty: float
    length_penalty: float


class ModelConfig(BaseModel, extra="forbid", validate_assignment=True):
    llama_path: Path
    beats_path: Path | GSPath | None = None
    beats_cfg: BeatsConfig
    ckpt: Path | GSPath | None = None
    freeze_beats: bool = True
    use_audio_Qformer: bool = True
    max_pooling: bool = False
    downsample_factor: int = 4
    freeze_audio_QFormer: bool = False
    window_level_Qformer: bool = True
    num_audio_query_token: int = 1
    second_per_window: float = 0.333333
    second_stride: float = 0.333333
    audio_llama_proj_model: Path | GSPath | None = None
    freeze_audio_llama_proj: bool = False
    device: str = "cuda"
    lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    flash_attn: Literal["eager", "flash_attention_2"] = "eager"
    prompt_template: str = ""
    max_txt_len: int = 128
    end_sym: str = "</s>"

    @field_validator("beats_path", "audio_llama_proj_model", "ckpt", mode="before")
    @classmethod
    def detect_gcs_path(cls, value: Any) -> Any:
        """Pydantic's automatic type conversion won't be able to deal with gs:// paths
        so we need to manually detect and convert them to GSPath objects _before_
        validation"""
        if value and is_gcs_path(value):
            return GSPath(str(value))
        else:
            return value

    @field_validator("ckpt", "audio_llama_proj_model", mode="before")
    @classmethod
    def legacy_empty_str(cls, value: Any) -> Any:
        """In some of our config files we use "" to indicate that we don't have
        a checkpoint. We've now switched to using None for this in the Config model but
        let's keep this validator for backwards compatibility so people don't have to
        change their configs"""
        if isinstance(value, str) and value == "":
            return None
        else:
            return value

    @classmethod
    def from_yaml(cls, yaml_file: str | os.PathLike) -> "ModelConfig":
        yaml_values = YamlConfigSettingsSource(cls, yaml_file=str(yaml_file))
        return cls.model_validate(yaml_values())


class Config(BaseSettings, extra="forbid", validate_assignment=True):
    model: ModelConfig
    run: RunConfig | None = None
    datasets: DatasetsConfig | None = None
    generate: GenerateConfig | None = None

    def pretty_print(self):
        print(self.model_dump_json(indent=4))

    @classmethod
    def from_sources(cls, yaml_file: str | Path, cli_args: list[str] = []) -> "Config":
        """Create a Config object from a YAML file and CLI arguments. If there are
        any conflicts, the CLI arguments will take precedence over the YAML file."""

        yaml_file = Path(yaml_file)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file {yaml_file} does not exist")

        yaml_values = YamlConfigSettingsSource(cls, yaml_file=yaml_file)
        cli_values = CliSettingsSource(cls, cli_parse_args=["--" + opt for opt in cli_args])
        final_values = deep_update(yaml_values(), cli_values())
        return cls.model_validate(final_values)

    def to_yaml(self, path: str | os.PathLike) -> None:
        save_config_as_yaml(self, path)


def save_config_as_yaml(data: BaseModel, filepath: str | os.PathLike) -> None:
    """
    Pydantic supports serializing/exporting models to various formats (dict, json, etc)
    but not to yaml. This function is a workaround for that limitation.
    """

    filepath = Path(filepath)

    if filepath.exists():
        raise FileExistsError(f"File {filepath} already exists")

    # The mode="json" is required because otherwise yaml.same_dump() can't deal with
    # Path|GSPath objects
    with filepath.open("w") as f:
        yaml.safe_dump(data.model_dump(mode="json"), f, sort_keys=False)
