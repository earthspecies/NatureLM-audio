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


"""
Mixing examples.
Can mix:
 - base: options-detection add: open-ended:
    Take all open-ended labels. Add them to the options. Add them to the labels.
- base: open-ended, add: open-ended
    Concatenate labels
"""

import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from NatureLM.utils import snr_scale, time_scale


def write_example_to_file(base_filename, audio, sr=16000, suffix="_output", save_dir="debug_outputs"):
    """
    Writes the audio tensor to a file for debugging or inspection purposes.

    Args:
        base_filename (str): The base name of the original file.
        audio (torch.Tensor or numpy.ndarray): The audio waveform to save.
        sr (int): Sampling rate of the audio (default: 16000 Hz).
        suffix (str): Optional suffix to append to the filename.
        save_dir (str): Directory where the files will be saved.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()  # Convert to numpy if necessary

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create the output file path
    filename = f"{os.path.splitext(base_filename)[0]}{suffix}.wav"
    output_path = os.path.join(save_dir, filename)

    try:
        # Write the audio to the file
        sf.write(output_path, audio, sr)
        print(f"Saved audio to {output_path}")
    except Exception as e:
        print(f"Failed to write audio to file: {e}")


# Example usage in your code
# write_example_to_file(os.path.basename(ann["path"]), audio, suffix="_ts")


def collater(samples):
    """Collate samples into a batch.

    Samples is a list of dictionaries, each containing the following keys:
    - raw_wav: a list of tensors containing the raw audio waveform
    - text: a list of strings containing the text
    - task: a list of strings containing the task
    - id: a list of strings containing the id
    - prompt: a list of strings containing the prompt
    - index: a list of integers containing the index

    The indiviudal audio waveforms will be stacked along the batch dimension for easier
    processing in the audio model. To keep which audio belongs to which sample, we add
    the audio_chunk_sizes key to the batch dictionary.
    """
    flat_raw_wav = []
    audio_chunk_sizes = []

    for s in samples:
        chunk_size = len(s["raw_wav"])
        audio_chunk_sizes.append(chunk_size)
        flat_raw_wav.extend(s["raw_wav"])
    # raw_wav = [torch.from_numpy(a) for a in flat_raw_wav]
    raw_wav = flat_raw_wav
    raw_wav_length = torch.tensor([len(a) for a in raw_wav])
    raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
    paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

    text = [s["text"] for s in samples]
    prompt = [s["prompt"] for s in samples]
    task = [s["task"] for s in samples]
    id = [s["id"] for s in samples]
    index = [s["index"] for s in samples]

    return {
        "raw_wav": raw_wav,
        "padding_mask": paddding_mask,
        "text": text,
        "task": task,
        "id": id,
        "prompt": prompt,
        "index": index,
        "audio_chunk_sizes": audio_chunk_sizes,
    }


class NatureLMDataset(Dataset):
    def __init__(
        self,
        ann_path: str | Path,
        *,
        max_length_seconds: int = 10,
        cropping: Literal["random", "start"] | None = "random",
        noise_prob: float = 0.0,
        noise_dirs: list[str] | list[Path] | None = None,
        low_snr: float = -5,
        high_snr: float = 20,
        time_scale_prob: float = 0.0,
        time_scale: float = 1.2,
        seed: int = 0,
        mixup_prob: float = 0.0,
        mixup_count: int = 3,
        use_augmentation: bool = False,
        mask_audio_prob: float = 0.0,
    ):
        super().__init__()

        ann_path = Path(ann_path)

        if not ann_path.exists():
            raise FileNotFoundError(f"Dataset file {ann_path} not found")

        try:
            with open(ann_path, "r") as f:
                data = json.load(f)
                self.annotation = data["annotation"]
        except (json.JSONDecodeError, KeyError):
            with open(ann_path, "r") as f:
                self.annotation = [json.loads(line) for line in f]

        #### mixup related variables
        ### hash table for tasks to sample the tasks faster
        self.tasks = defaultdict(list)
        for i, ann in enumerate(self.annotation):
            if "task" in ann and "text" in ann and ann["text"] != "None" and "path" in ann:
                self.tasks[ann["task"]].append(i)

        self.mixup_tasks = {
            task: []
            for task in self.tasks.keys()
            if task.endswith("simple-detection")
            or task.endswith("multiple-detection")  # Add more tasks after validating prompt mixing.
            or task.endswith("sci-detection-random")
            or task.endswith("common-detection-random")
        }
        for k in self.mixup_tasks.keys():
            # whichever the base, only mix open-ended tasks.
            if "sci-" in k:
                self.mixup_tasks[k] = [
                    task
                    for task in self.mixup_tasks.keys()
                    if task.endswith("sci-simple-detection") or task.endswith("sci-multiple-detection")
                ]
            elif "common-" in k:
                self.mixup_tasks[k] = [
                    task
                    for task in self.mixup_tasks.keys()
                    if task.endswith("common-simple-detection") or task.endswith("common-multiple-detection")
                ]
            else:
                self.mixup_tasks[k] = [task for task in self.mixup_tasks.keys() if "common-" in task]

        # print("num annotations", len(self.annotation))
        # print("annotation 0", self.annotation[0])
        # self.annotation = [a for a in self.annotation if "task" in a and "detection" not in a["task"]] # no detection... :(
        self.max_length_seconds = max_length_seconds
        self.cropping = cropping
        self.use_augmentation = use_augmentation

        ### noise augmentation
        self.rng = random.Random(seed)
        self.rngnp = np.random.default_rng(seed=seed)
        self.noise_dirs = noise_dirs
        self.noise_prob = noise_prob
        self.noise_files = []
        self.low_snr = low_snr
        self.high_snr = high_snr
        self.mask_audio_prob = mask_audio_prob
        if noise_dirs is not None and len(self.noise_dirs) > 0 and self.use_augmentation:
            for noise_dir in noise_dirs:
                noise_from_dir = glob.glob(os.path.join(noise_dir, "*.wav"))
                if len(noise_from_dir) < 3000:
                    noise_from_dir = noise_from_dir * 3
                print("noise files from dir", noise_dir, len(noise_from_dir))
                self.noise_files.extend(noise_from_dir)

        ### mixup augmentation
        self.mixup_prob = mixup_prob
        self.mixup_count = mixup_count
        # ### time scale augmentation
        self.time_scale = time_scale
        self.time_scale_prob = time_scale_prob
        # tasks = set([annotation["task"] if "task" in annotation else "empty" for annotation in self.annotation])
        print(":::all tasks:::", self.tasks.keys())
        print("num examples", len(self.annotation))

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return collater(samples)

    def load_audio(self, audio_path, shift_allowed: bool, noise_allowed: bool):
        audio, sr = sf.read(audio_path)
        # assert sr == 16000
        if sr != 16000:
            print("other sr!", sr, audio_path)
        if len(audio.shape) == 2:  # stereo to mono
            audio = audio.mean(axis=1)

        ### time scale augmentation
        if self.use_augmentation and self.rng.random() < self.time_scale_prob and self.time_scale > 0 and shift_allowed:
            # print(f"{index} scaling audio")
            # write_example_to_file(os.path.basename(ann["path"]), audio[: sr * self.max_length_seconds] )
            audio = time_scale(torch.tensor(audio), scale=self.time_scale, rngnp=self.rngnp).numpy()
            # write_example_to_file(os.path.basename(ann["path"]), audio[: sr * self.max_length_seconds] , suffix='_ts')

        # Randomly crop a max_length_seconds window if audio is longer than 10 seconds
        if len(audio) > sr * self.max_length_seconds and self.cropping == "random":
            max_start = len(audio) - sr * self.max_length_seconds
            start = random.randint(0, max_start)
            audio = audio[start : start + sr * self.max_length_seconds]
        else:  # no random cropping
            audio = audio[: sr * self.max_length_seconds]  # Truncate audio to at most max_length_seconds

        ### noise augmentation
        audio = torch.tensor(audio)
        ### noise augmentation
        if (
            self.use_augmentation
            and self.rng.random() < self.noise_prob
            and len(self.noise_files) > 0
            and noise_allowed
        ):
            # write_example_to_file(os.path.basename(ann["path"]), audio)
            # print(f"{index} adding noise")
            noise_file = self.rng.choice(self.noise_files)
            if not os.path.exists(noise_file):
                print(f"Warning: noise file {noise_file} does not exist")
            else:
                noise_audio, noise_sr = sf.read(noise_file)
                assert noise_sr == 16000
                if len(noise_audio.shape) == 2:
                    noise_audio = noise_audio.mean(axis=1)

                noise_audio = torch.tensor(noise_audio)

                ### repeat or trim to the audio size
                if len(audio) > len(noise_audio):
                    if len(noise_audio) == 0:
                        print(
                            "----- Warning: Noise audio length is zero. ---------- ",
                            noise_file,
                        )
                        # Option 1: Skip noise augmentation by setting noise_audio to zero
                        noise_audio = torch.zeros_like(audio)
                    else:
                        nrepeats = int(np.maximum(2, np.ceil(len(audio) / len(noise_audio))))
                        noise_audio = noise_audio.repeat(nrepeats)
                ### Randomly crop the noise file if it is too long
                if len(noise_audio) > len(audio):
                    max_start = len(noise_audio) - len(audio)
                    start = random.randint(0, max_start)
                    noise_audio = noise_audio[start : start + len(audio)]

                ### remix with specified snr
                snr = self.rngnp.uniform(self.low_snr, self.high_snr)
                snr = torch.tensor([snr])
                noise_audio = snr_scale(audio, noise_audio, snr)
                audio = audio + noise_audio

                # write_example_to_file(os.path.basename(audio_path), audio, suffix='_noise')
            if len(audio) > self.max_length_seconds * sr:
                print("long audio", len(audio), len(noise_audio))
                audio = audio[: self.max_length_seconds * sr]

        # pad all audios to max_len_seconds in _getitem_ to ensure no padding inconsistencies.
        if len(audio) < sr * self.max_length_seconds:
            pad_size = sr * self.max_length_seconds - len(audio)
            audio = torch.nn.functional.pad(audio, (0, pad_size))

        audio = torch.clamp(audio, -1.0, 1.0)

        return audio

    def _mix_labels(self, text, text_to_mix):
        """
        Given two comma-separated label strings (e.g., "gorilla, zebra"),
        combine them without introducing duplicates. If either is "None",
        return the other as-is (unless both are "None").
        """
        # If `text_to_mix` is explicitly "None", just return `text`.
        if text_to_mix == "None":
            return text

        # If `text` is explicitly "None", just return `text_to_mix`.
        if text == "None":
            return text_to_mix

        # Split both strings by comma, stripping whitespace
        text_list = [item.strip() for item in text.split(",") if item.strip()]
        text_to_mix_list = [item.strip() for item in text_to_mix.split(",") if item.strip()]

        # Deduplicate: add only new items from text_to_mix_list
        combined_set = set(text_list)
        for item in text_to_mix_list:
            if item not in combined_set:
                text_list.append(item)
                combined_set.add(item)

        # If there's nothing left after deduplication, return "None".
        if not text_list:
            return "None"

        # Rejoin them into a comma-separated string
        return ", ".join(text_list)

    def _mix_prompts(self, text, text_to_mix, prompt):
        """
        If the prompt is in the form:
            "Which of these, if any, are present in the audio recording? option1, option2, ..."

        1. Parse out the question (before '?') and the list of prompt choices (after '?').
        2. Convert both `text` and `text_to_mix` into lists, checking for items not in the prompt.
        3. Append any missing answers to the prompt choices.
        4. Shuffle the choices.
        5. Reassemble and return the new prompt.

        If the prompt does not follow the expected structure, it is returned unmodified.
        """
        # Split into two parts: question + choices
        splitted = prompt.split("?")
        if len(splitted) != 2:
            # If we don't have exactly one question mark segment, just return the original prompt
            return prompt

        question = splitted[0].strip()
        potential_choices_str = splitted[1].strip()

        # Split the prompt choices
        if not potential_choices_str:
            prompt_choices = []
        else:
            prompt_choices = [c.strip() for c in potential_choices_str.split(",") if c.strip()]

        # Parse `text`
        text_list = [item.strip() for item in text.split(",") if item.strip()]

        # Parse `text_to_mix`
        text_to_mix_list = [item.strip() for item in text_to_mix.split(",") if item.strip()]

        # Add any new items from text_list to the prompt
        for item in text_list:
            if item not in prompt_choices:
                prompt_choices.append(item)

        # Add any new items from text_to_mix_list to the prompt
        for item in text_to_mix_list:
            if item not in prompt_choices:
                prompt_choices.append(item)

        # Shuffle consistently with self.rng
        self.rng.shuffle(prompt_choices)

        # Reassemble
        new_prompt = question + "? " + ", ".join(prompt_choices)
        return new_prompt

    def _apply_mixup(self, prompt, audio, text, task, filename=None):
        # mixup_applied = False
        if (
            self.use_augmentation and self.rng.random() < self.mixup_prob and task in self.mixup_tasks
            # and text != "None" # Allow complex 'None' examples.
        ):
            # write_example_to_file(os.path.basename(ann["path"]), audio)
            # print(f"{index} mixing up")
            mixup_indices = []
            for pair_task in self.mixup_tasks[task]:
                mixup_indices.extend(self.tasks[pair_task])
            # mixup_indices = mixup_indices.remove(index)

            if len(mixup_indices) == 0:
                print("No mixup partner found")
            else:
                ### choose n_mixup random partners
                n_mixup = self.rng.randint(1, self.mixup_count)
                mixup_indices = self.rng.sample(mixup_indices, n_mixup)
                # print(f"Mixing up with indices {mixup_indices}")
                for mixup_index in mixup_indices:
                    mixup_ann = self.annotation[mixup_index]
                    mixup_audio, _ = sf.read(mixup_ann["path"])
                    if len(mixup_audio.shape) == 2:
                        mixup_audio = mixup_audio.mean(axis=1)
                    mixup_audio = mixup_audio[: len(audio)]
                    if len(mixup_audio) < len(audio):
                        pad_size = len(audio) - len(mixup_audio)
                        mixup_audio = np.pad(mixup_audio, (0, pad_size), mode="constant")
                    mixup_audio = torch.from_numpy(mixup_audio).float()
                    lam = np.clip(self.rngnp.beta(1.0, 1.0), 0.1, 0.8)

                    # Mix the raw_wav
                    audio = lam * audio + (1 - lam) * mixup_audio

                    ### Mix the prompts if the labels are given in prompts
                    if text in prompt:
                        prompt = self._mix_prompts(text, mixup_ann["text"], prompt)

                    ### Mix the labels
                    text = self._mix_labels(text, mixup_ann["text"])

                # mixup_applied = True

        # DEBUG: If mixup was actually applied, save the final audio
        # if mixup_applied and filename is not None:
        #     # Just add a suffix to the original filename to indicate mixup
        #     base_filename = os.path.basename(filename)
        #     write_example_to_file(
        #         base_filename=base_filename,
        #         audio=audio,
        #         sr=16000,
        #         suffix="_mixup",
        #         save_dir="mixup_outputs"
        #     )
        #     print(f"mixup for {filename}::: prompt {prompt} label {text}")

        return prompt, audio, text

    def _load_noise(self, shift_allowed: bool):
        noise_file = self.rng.choice(self.noise_files)
        noise_audio, noise_sr = sf.read(noise_file)
        assert noise_sr == 16000, f"Expected noise sample rate 16000, got {noise_sr}"
        if len(noise_audio.shape) == 2:
            noise_audio = noise_audio.mean(axis=1)

        # Time scale augmentation if applicable
        if self.use_augmentation and self.rng.random() < self.time_scale_prob and self.time_scale > 0 and shift_allowed:
            noise_audio = time_scale(torch.tensor(noise_audio), scale=self.time_scale, rngnp=self.rngnp).numpy()

        # Randomly crop or pad to match max_length_seconds
        if len(noise_audio) > self.max_length_seconds * 16000 and self.cropping == "random":
            max_start = len(noise_audio) - self.max_length_seconds * 16000
            start = random.randint(0, max_start)
            noise_audio = noise_audio[start : start + self.max_length_seconds * 16000]
        else:
            noise_audio = noise_audio[: self.max_length_seconds * 16000]

        # Pad if needed
        if len(noise_audio) < self.max_length_seconds * 16000:
            pad_size = self.max_length_seconds * 16000 - len(noise_audio)
            noise_audio = np.pad(noise_audio, (0, pad_size), mode="constant")

        noise_audio = torch.tensor(noise_audio).float()
        noise_audio = torch.clamp(noise_audio, -1.0, 1.0)
        return noise_audio

    def __getitem__(self, index):
        ann = self.annotation[index]
        # print("loading audio::", ann)
        shift_allowed = "pitch" not in ann.get("task", "")
        noise_allowed = (
            "/A/" not in ann.get("path", "")
            and "-qa" not in ann.get("task", "")
            and "icl" not in ann.get("task", "")
            and "caption" not in ann.get("task", "")
            and "animal-instructions" not in ann.get("task", "")
        )

        task = ann.get("task", "asr")
        text = ann["text"]
        prompt = ann["prompt"]

        replace_with_noise = (
            self.use_augmentation
            and task.endswith("detection")
            and self.rng.random() < self.mask_audio_prob
            and len(self.noise_files) > 0
        )

        if replace_with_noise:
            # Replace audio with noise
            audio = self._load_noise(shift_allowed)
            audios = [audio]
            text = "None"

        else:
            if "path" in ann and ann["path"] is not None:
                audio = self.load_audio(ann["path"], shift_allowed, noise_allowed)
                audios = [audio]
            else:
                audios = [self.load_audio(p, shift_allowed, noise_allowed) for p in ann["files"]]

            if len(audios) == 1:
                prompt, mixed_audio, text = self._apply_mixup(prompt, audio, text, task, filename=ann["path"])
                audios = [mixed_audio]

        return {
            "raw_wav": audios,
            "text": text,
            "task": task,
            "id": ann.get("path") or ";".join(ann["files"]),
            "prompt": prompt,
            "index": index,  # track which element for eval output
            "ann": ann,  # Include annotation for mixup
        }


if __name__ == "__main__":
    dataset = NatureLMDataset(
        ann_path="/home/ubuntu/foundation-model-storage/foundation-model-data/data/compiled-datasets/v1/s2_eval_valid.jsonl",
        noise_dirs=["resource/audio_demo"],
        max_length_seconds=10,
        use_augmentation=True,
        mixup_prob=1.0,  # For demonstration, force mixup if possible
        mixup_count=2,  # Up to 2 mixup partners
        mask_audio_prob=0.2,
        seed=42,
        noise_prob=0.5,
    )

    # Process just a few to see the saved mixups
    for i in range(300):
        sample = dataset[i]
        # print("Final text:", sample["text"])
        # print("Final prompt:", sample["prompt"])
        # print("-" * 40)
    print("Done! Look in 'debug_outputs' folder for saved mixup files.")
