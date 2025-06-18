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

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import resampy
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, DistributedSampler

from NatureLM.dist_utils import get_rank, get_world_size
from NatureLM.storage_utils import GSPath, is_gcs_path

logger = logging.getLogger(__name__)


TARGET_SAMPLE_RATE = 16_000


def snr_scale(clean, noise, snr):
    # Ensure both clean and noise have the same length
    assert clean.shape == noise.shape, "Clean and noise must have the same shape."

    # Compute power (mean squared amplitude)
    power_signal = torch.mean(clean**2)
    power_noise = torch.mean(noise**2)

    # Prevent division by zero
    epsilon = 1e-10
    power_noise = torch.clamp(power_noise, min=epsilon)

    # Calculate desired noise power based on SNR
    desired_noise_power = power_signal / (10 ** (snr / 10))

    # Scale noise to achieve the desired noise power
    scale = torch.sqrt(desired_noise_power / power_noise)
    scaled_noise = scale * noise

    return scaled_noise


def time_scale(signal, scale=2.0, rngnp=None, seed=42):
    if rngnp is None:
        rngnp = np.random.default_rng(seed=seed)
    scaling = np.power(scale, rngnp.uniform(-1, 1))
    output_size = int(signal.shape[-1] * scaling)
    ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)
    ref1 = ref.clone().type(torch.int64)
    ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
    r = ref - ref1.type(ref.type())
    scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r

    ## trim or zero pad to torche original size
    if scaled_signal.shape[-1] > signal.shape[-1]:
        nframes_offset = (scaled_signal.shape[-1] - signal.shape[-1]) // 2
        scaled_signal = scaled_signal[..., nframes_offset : nframes_offset + signal.shape[-1]]
    else:
        nframes_diff = signal.shape[-1] - scaled_signal.shape[-1]
        pad_left = int(np.random.uniform() * nframes_diff)
        pad_right = nframes_diff - pad_left
        scaled_signal = F.pad(input=scaled_signal, pad=(pad_left, pad_right), mode="constant", value=0)
    return scaled_signal


def mel_frequencies(n_mels, fmin, fmax):
    def _hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def _mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    low = _hz_to_mel(fmin)
    high = _hz_to_mel(fmax)

    mels = np.linspace(low, high, n_mels)

    return _mel_to_hz(mels)


def now_as_str() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


def get_dataloader(dataset, config, is_train=True, use_distributed=True):
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train, num_replicas=get_world_size(), rank=get_rank())
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train if is_train else config.batch_size_eval,
        num_workers=config.num_workers,
        pin_memory=False,
        sampler=sampler,
        shuffle=sampler is None and is_train,
        collate_fn=dataset.collater,
        drop_last=is_train,
    )

    if is_train:
        loader = IterLoader(loader, use_distributed=use_distributed)

    return loader


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_device(sample, device):
    def _move_to_device(tensor):
        return tensor.to(device)

    return apply_to_sample(_move_to_device, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    # TODO fp16 support

    return samples


def prepare_sample_dist(samples, device):
    samples = move_to_device(samples, device)

    # TODO fp16 support

    return samples


class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


def prepare_one_sample(wav_path: str, wav_processor=None, cuda_enabled=True) -> dict:
    """Prepare a single sample for inference.

    Args:
        wav_path: Path to the audio file.
        wav_processor: A function to process the audio file.
        cuda_enabled: Whether to move the sample to the GPU.
    """
    audio, sr = sf.read(wav_path)
    if len(audio.shape) == 2:  # stereo to mono
        audio = audio.mean(axis=1)
    if len(audio) < sr:  # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    audio = audio[: sr * 10]  # truncate audio to at most 10s

    # spectrogram = wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]
    print("audio shape", audio.shape)

    audio_t = torch.tensor(audio).unsqueeze(0)
    audio_t = torchaudio.functional.resample(audio_t, sr, TARGET_SAMPLE_RATE)
    print("audio shape after resample", audio_t.shape)

    samples = {
        "raw_wav": audio_t,
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        "audio_chunk_sizes": [1],
    }
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    return samples


def prepare_one_sample_waveform(audio, cuda_enabled=True, sr=16000):
    print("shape", audio.shape)
    if len(audio.shape) == 2:  # stereo to mono
        print("converting stereo to mono?")
        audio = audio.mean(axis=1)
    if len(audio) < sr:  # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    audio = audio[: sr * 10]  # truncate audio to at most 30s

    samples = {
        "raw_wav": torch.tensor(audio).unsqueeze(0).type(torch.DoubleTensor),
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
    }
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    return samples


def prepare_sample_waveforms(audio_paths, cuda_enabled=True, sr=TARGET_SAMPLE_RATE, max_length_seconds=10):
    batch_len = sr  # minimum length of audio
    audios = []
    for audio_path in audio_paths:
        audio, loaded_sr = sf.read(audio_path)
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        audio = audio[: loaded_sr * 10]
        audio = resampy.resample(audio, loaded_sr, sr)
        audio = torch.from_numpy(audio)

        if len(audio) < sr * max_length_seconds:
            pad_size = sr * max_length_seconds - len(audio)
            audio = torch.nn.functional.pad(audio, (0, pad_size))
        audio = torch.clamp(audio, -1.0, 1.0)
        if len(audio) > batch_len:
            batch_len = len(audio)
        audios.append(audio)
    padding_mask = torch.zeros((len(audios), batch_len), dtype=torch.bool)
    for i in range(len(audios)):
        if len(audios[i]) < batch_len:
            pad_len = batch_len - len(audios[i])
            sil = torch.zeros(pad_len, dtype=torch.float32)
            audios[i] = torch.cat((audios[i], sil), dim=0)
            padding_mask[i, len(audios[i]) :] = True
    audios = torch.stack(audios, dim=0)

    samples = {
        "raw_wav": audios,
        "padding_mask": padding_mask,
        "audio_chunk_sizes": [len(audio_paths)],
    }
    if cuda_enabled:
        samples = move_to_device(samples, "cuda")

    return samples


def generate_sample_batches(
    audio_path,
    cuda_enabled: bool = True,
    sr: int = TARGET_SAMPLE_RATE,
    chunk_len: int = 10,
    hop_len: int = 5,
    batch_size: int = 4,
):
    audio, loaded_sr = sf.read(audio_path)
    if len(audio.shape) == 2:  # stereo to mono
        audio = audio.mean(axis=1)
    audio = torchaudio.functional.resample(torch.from_numpy(audio), loaded_sr, sr)
    hop_len = hop_len * sr
    chunk_len = max(len(audio), chunk_len * sr)
    chunks = []

    for i in range(0, len(audio), hop_len):
        chunk = audio[i : i + chunk_len]
        if len(chunk) < chunk_len:
            break
        chunks.append(chunk)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        padding_mask = torch.zeros((len(batch), sr * chunk_len), dtype=torch.bool)
        batch = torch.stack(batch, dim=0)
        samples = {
            "raw_wav": batch,
            "padding_mask": padding_mask,
            "audio_chunk_sizes": [1 for _ in range(len(batch))],
        }
        if cuda_enabled:
            samples = move_to_device(samples, "cuda")
        yield samples


def prepare_samples_for_detection(samples, prompt, label):
    prompts = [prompt for i in range(len(samples["raw_wav"]))]
    labels = [label for i in range(len(samples["raw_wav"]))]
    task = ["detection" for i in range(len(samples["raw_wav"]))]
    samples["prompt"] = prompts
    samples["text"] = labels
    samples["task"] = task
    return samples


def universal_torch_load(
    f: str | os.PathLike,
    *,
    cache_mode: Literal["none", "use", "force"] = "none",
    **kwargs,
) -> Any:
    """
    Wrapper function for torch.load that can handle GCS paths.

    This function provides a convenient way to load PyTorch objects from both local and
    Google Cloud Storage (GCS) paths. For GCS paths, it can optionally caches the
    downloaded files locally to avoid repeated downloads.

    The cache location is determined by:
    1. The ESP_CACHE_HOME environment variable if set
    2. Otherwise defaults to ~/.cache/esp/

    Args:
        f: File-like object, string or PathLike object.
           Can be a local path or a GCS path (starting with 'gs://').
        cache_mode (str, optional): Cache mode for GCS files. Options are:
            "none": No caching (use bucket directly)
            "use": Use cache if available, download if not
            "force": Force redownload even if cache exists
            Defaults to "none".
        **kwargs: Additional keyword arguments passed to torch.load().

    Returns:
        The object loaded from the file using torch.load.

    Raises:
        IsADirectoryError: If the GCS path points to a directory instead of a file.
        FileNotFoundError: If the local file does not exist.
    """

    if is_gcs_path(f):
        gs_path = GSPath(str(f))
        if gs_path.is_dir():
            raise IsADirectoryError(f"Cannot load a directory: {f}")

        if cache_mode in ["use", "force"]:
            if "ESP_CACHE_HOME" in os.environ:
                cache_path = Path(os.environ["ESP_CACHE_HOME"]) / gs_path.name
            else:
                cache_path = Path.home() / ".cache" / "esp" / gs_path.name

            if not cache_path.exists() or cache_mode == "force":
                logger.info(
                    f"{'Force downloading' if cache_mode == 'force' else 'Cache file does not exist, downloading'} to {cache_path}..."
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                gs_path.download_to(cache_path)
            else:
                logger.debug(f"Found {cache_path}, using local cache.")
            f = cache_path
        else:
            f = gs_path
    else:
        f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"File does not exist: {f}")

    with open(f, "rb") as opened_file:
        return torch.load(opened_file, **kwargs)
