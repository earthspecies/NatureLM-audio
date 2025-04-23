"""Module contains the audio and text processor for NatureLM-audio inference and evaluation"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import resampy
import soundfile as sf
import torch


@dataclass
class NatureLMAudioProcessor:
    """Preprocess samples to make them ready for NatureLM-audio inference.

    Arguments
    ---------
    naturelm_sample_rate : int
        The sample rate of the NatureLM model
    max_length_seconds : int
        The maximum length of audio in seconds
    audio_token_placeholder : str
        The placeholder for the audio token in the instruction
    prompt_template : str
        The template for the prompt. The instruction or query from the user is inserted in the placeholder at {prompt}


    Examples
    --------
    >>> processor = NatureLMAudioProcessor()
    >>> audios = [np.random.rand(32000), np.random.rand(32000)]
    >>> instructions = ["What is the weather today?", "What is the time now?"]
    >>> input_sample_rates = [32000, 32000]
    >>> audios, instructions = processor(audios, instructions, input_sample_rates)
    >>> audios.shape == (2, 160000)
    True
    >>> "<Audio><AudioHere></Audio> " in instructions[0]
    True
    >>> "<|start_header_id|>user<|end_header_id|>" in instructions[0]
    True
    """

    sample_rate: int = 16000
    max_length_seconds: int = 10
    audio_token_placeholder: str = "<Audio><AudioHere></Audio> "
    prompt_template: str = "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def prepare_audio(self, audio: list[float] | np.ndarray | os.PathLike, input_sr: int = None) -> torch.Tensor:
        """Prepare an audio array or file path for inference"""
        if isinstance(audio, str | os.PathLike):
            audio, sr = sf.read(audio)
            input_sr = sr
        elif isinstance(audio, list):
            audio = np.array(audio)

        assert isinstance(audio, np.ndarray), "Audio not a numpy array"

        # Convert stereo to mono
        if len(audio.shape) == 2:
            # find the smaller axis as channel dim to avg over (like (2, T) or (T, 2), 2 = channel dim
            axis_to_average = int(np.argmin(audio.shape))
            audio = audio.mean(axis=axis_to_average)

        # Resample
        if input_sr is not None and input_sr != self.sample_rate:
            # audio = torchaudio.functional.resample(
            #     torch.from_numpy(audio), orig_freq=input_sr, new_freq=self.sample_rate
            # )
            audio = resampy.resample(audio, input_sr, self.sample_rate)
            audio = torch.from_numpy(audio.squeeze())
        else:
            audio = torch.from_numpy(audio)

        # Truncate audio to at most max_length_seconds
        audio = audio[: self.sample_rate * self.max_length_seconds]

        # Pad to max_length_seconds if short
        if len(audio) < self.sample_rate * self.max_length_seconds:
            pad_size = self.sample_rate * self.max_length_seconds - len(audio)
            audio = torch.nn.functional.pad(audio, (0, pad_size))

        # Clamp
        audio = torch.clamp(audio, -1.0, 1.0)

        return audio.squeeze()

    def prepare_instruction(self, instruction: str) -> str:
        """Add the audio token placeholder to the instruction and format it
        according to the llama tokenizer.
        """
        if self.audio_token_placeholder not in instruction:
            instruction = self.audio_token_placeholder + instruction
        instruction = self.prompt_template.format(prompt=instruction.strip())

        return instruction

    def __call__(
        self,
        audios: list[list[float] | np.ndarray] | list[str | os.PathLike],
        instructions: list[str],
        input_sample_rates: list[int],
    ) -> tuple[torch.Tensor, list[str]]:
        """Prepare audios and instructions for inference

        Arguments
        ---------
        audios : list[list[float] | np.ndarray] | list[str | os.PathLike]
            The audio samples or file paths
        instructions : list[str]
            The instructions or queries
        input_sample_rates : list[int]
            The sample rates of the input audio samples

        Returns
        -------
        tuple[torch.Tensor, list[str]]
            The prepared audios and instructions
        """
        audios = torch.stack(
            [self.prepare_audio(audio, input_sr) for audio, input_sr in zip(audios, input_sample_rates)]
        )
        instructions = [self.prepare_instruction(instruction) for instruction in instructions]

        return audios, instructions


@dataclass
class NatureLMAudioEvalProcessor(NatureLMAudioProcessor):
    """Preprocess samples to make them ready for NatureLM-audio evaluation on BEANS-Zero dataset.
    This requires a few additional parameters compared to the NatureLMAudioProcessor.

    Arguments
    ---------
    naturelm_sample_rate : int
        The sample rate of the NatureLM model
    max_length_seconds : int
        The maximum length of audio in seconds
    audio_token_placeholder : str
        The placeholder for the audio token in the instruction
    prompt_template : str
        The template for the prompt. The instruction or query from the user is inserted in the placeholder at {prompt}

    dataset_name : list[str]
        The name of the dataset being processed
    true_labels : list[str]
        The true labels or expected outputs for the samples.
    task: str
        The task for the dataset. Can be 'detection', 'captioning', or 'classification'
    threshold_too_many_detection_labels : int
        The threshold for the number of labels in the dataset to switch to a detection prompt. Default is 8.


    Examples
    --------
    >>> processor = NatureLMAudioEvalProcessor(task="detection", true_labels=["dog", "cat", "bird", "None", "mouse", "elephant", "lion", "tiger", "bear"])
    >>> audios = [np.random.rand(32000), np.random.rand(32000)]
    >>> instructions = ["What is the weather today?", "What is the time now?"]
    >>> input_sample_rates = [32000, 32000]
    >>> audios, instructions = processor(audios, instructions, input_sample_rates)
    >>> audios.shape == (2, 160000)
    True
    >>> "<Audio><AudioHere></Audio> " in instructions[0]
    True
    >>> "<|start_header_id|>user<|end_header_id|>" in instructions[0]
    True
    >>> "What are the common names" in instructions[0]
    True
    """

    dataset_name: str = "beans-zero"
    true_labels: list[str] = field(default_factory=lambda _: [])
    task: str = "detection"

    threshold_too_many_detection_labels: int = 8

    def __post_init__(self):
        self.detection_prompt: str = (
            "<Audio><AudioHere></Audio> What are the common names for the species in the audio, if any?"
        )

        # find the unique labels in the dataset
        self.dataset_labels = set(self.true_labels)
        if self.task == "detection":
            self.dataset_labels.add("None")
        if self.task == "captioning":
            self.dataset_labels = set()

    def prepare_instruction(self, instruction: str) -> str:
        """Add the audio token placeholder to the instruction and format it"""
        if self.task == "detection" and len(self.dataset_labels) > self.threshold_too_many_detection_labels:
            instruction = self.detection_prompt

        if self.audio_token_placeholder not in instruction:
            instruction = self.audio_token_placeholder + instruction

        instruction = self.prompt_template.format(prompt=instruction.strip())

        return instruction


class NatureLMInferenceDataset(torch.utils.data.Dataset):
    """A pytorch dataset for batched inference with NatureLM-audio

    TODO: currently, if the batch contains very different prompts the model doesnt work well.

    Arguments
    ---------
    ds : datasets.Dataset
        The huggingface dataset containing the samples

    Examples
    --------
    TODO: Add examples
    """

    def __init__(self, ds, processor):
        self.ds = ds
        self.processor = processor

    def __getitem__(self, idx):
        sample = self.ds[idx]
        input_sample_rate = json.loads(sample["metadata"])["sample_rate"]
        audio_tensor = self.processor.prepare_audio(sample["audio"], input_sample_rate)

        instruction = self.processor.prepare_instruction(sample["instruction"])
        return {
            "raw_wav": audio_tensor,
            "text": "",
            "task": sample["task"],
            "audio_chunk_sizes": len(audio_tensor),
            "index": idx,
            "id": sample["id"],
            "prompt": instruction,
            "label": sample["output"],
        }

    def __len__(self):
        return len(self.ds)


def collater(samples: list[dict]) -> dict:
    """Collate samples into a batch.

    Samples is a list of dictionaries, each containing the following keys:
    - raw_wav: a list of tensors containing the raw audio waveform
    - text: a list of strings containing the text
    - task: a list of strings containing the task
    - id: a list of strings containing the id
    - prompt: a list of strings containing the prompt
    - index: a list of integers containing the index
    - audio_chunk_sizes: a list of integers containing the size of each audio chunk

    The indiviudal audio waveforms will be stacked along the batch dimension for easier
    processing in the audio model. To keep which audio belongs to which sample, we add
    the audio_chunk_sizes key to the batch dictionary.
    """
    raw_wav = torch.stack([s["raw_wav"] for s in samples])
    paddding_mask = torch.zeros_like(raw_wav).to(torch.bool)

    text = [s["text"] for s in samples]
    prompt = [s["prompt"] for s in samples]
    task = [s["task"] for s in samples]
    id = [s["id"] for s in samples]
    index = [s["index"] for s in samples]
    label = [s["label"] for s in samples]

    return {
        "raw_wav": raw_wav,
        "padding_mask": paddding_mask,
        "text": text,
        "task": task,
        "id": id,
        "prompt": prompt,
        "index": index,
        "audio_chunk_sizes": 1,
        "label": label,
    }
