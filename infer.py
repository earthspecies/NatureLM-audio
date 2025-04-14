"""Run NatureLM-audio over a set of audio files paths or a directory with audio files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from NatureLM.config import Config
from NatureLM.models import NatureLM
from NatureLM.processors import NatureLMAudioProcessor
from NatureLM.utils import move_to_device

MAX_LENGTH_SECONDS = 10
MIN_CHUNK_LENGTH_SECONDS = 0.5
SAMPLE_RATE = 16000  # Assuming the model uses a sample rate of 16kHz
_AUDIO_FILE_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg"]  # Add other audio file formats as needed
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model_and_config(cfg_path: str | Path = "configs/inference.yml") -> tuple[NatureLM, Config]:
    """Load the NatureLM model and configuration.
    Returns:
        tuple: The loaded model and configuration.
    """
    model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
    model = model.to(DEVICE).eval()
    model.llama_tokenizer.pad_token_id = model.llama_tokenizer.eos_token_id
    model.llama_model.generation_config.pad_token_id = model.llama_tokenizer.pad_token_id

    cfg = Config.from_sources(cfg_path)
    return model, cfg


def output_template(model_output: str, start_time: float, end_time: float) -> str:
    """Format the output of the model."""
    return f"#{start_time:.2f}s - {end_time:.2f}s#: {model_output}\n"


def sliding_window_inference(
    audio_path: str | Path,
    query: str,
    processor: NatureLMAudioProcessor,
    model: NatureLM,
    cfg: Config,
    window_length_seconds: float = 10.0,
    hop_length_seconds: float = 10.0,
) -> str:
    """Run inference on a long audio file using sliding window approach.

    Args:
        audio_path (str | Path): Path to the audio file.
        query (str): Query for the model.
        processor (NatureLMAudioProcessor): Audio processor.
        model (NatureLM): NatureLM model.
        cfg (Config): Model configuration.
        window_length_seconds (float): Length of the sliding window in seconds.
        hop_length_seconds (float): Hop length for the sliding window in seconds.

    Returns:
        str: The output of the model.
    """
    audio_array, input_sr = sf.read(str(audio_path))
    audio_array = audio_array.squeeze()
    if audio_array.ndim > 1:
        axis_to_average = int(np.argmin(audio_array.shape))
        audio_array = audio_array.mean(axis=axis_to_average)
        audio_array = audio_array.squeeze()

    start = 0
    stride = int(hop_length_seconds * input_sr)
    window_length = int(window_length_seconds * input_sr)

    output = ""
    while True:
        chunk = audio_array[start : start + window_length]
        if chunk.shape[-1] < int(MIN_CHUNK_LENGTH_SECONDS * input_sr):
            break

        # Resamples, pads, truncates and creates torch Tensor
        audio_tensor, prompt_list = processor([chunk], [query], [input_sr])
        torch.cuda.empty_cache()

        input_to_model = {
            "raw_wav": audio_tensor,
            "prompt": prompt_list[0],
            "audio_chunk_sizes": 1,
            "padding_mask": torch.zeros_like(audio_tensor).to(torch.bool),
        }
        input_to_model = move_to_device(input_to_model, DEVICE)

        # generate
        prediction: str = model.generate(input_to_model, cfg.generate, prompt_list)[0]

        # Post-process the prediction
        prediction = output_template(prediction, start / input_sr, (start + window_length) / input_sr)
        output += prediction

        # Move the window
        start += stride

        if start + window_length > audio_array.shape[-1]:
            break

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run NatureLM-audio inference")
    parser.add_argument(
        "-a", "--audio", type=str, required=True, help="Path to an audio file or a directory containing audio files"
    )
    parser.add_argument("-q", "--query", type=str, required=True, help="Query for the model")
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="configs/inference.yml",
        help="Path to the configuration file for the model",
    )
    parser.add_argument("--output_path", type=str, default="inference_output.jsonl", help="Output path for the results")
    parser.add_argument(
        "--window_length_seconds", type=float, default=10.0, help="Length of the sliding window in seconds"
    )
    parser.add_argument(
        "--hop_length_seconds", type=float, default=10.0, help="Hop length for the sliding window in seconds"
    )
    args = parser.parse_args()

    return args


def main(
    cfg_path: str | Path,
    audio_path: str | Path,
    query: str,
    output_path: str,
    window_length_seconds: float,
    hop_length_seconds: float,
) -> None:
    """Main function to run the NatureLM-audio inference script.
    It takes command line arguments for audio file path, query, output path,
    window length, and hop length. It processes the audio files and saves the
    results to a CSV file.

    Args:
        cfg_path (str | Path): Path to the configuration file.
        audio_path (str | Path): Path to the audio file or directory.
        query (str): Query for the model.
        output_path (str): Path to save the output results.
        window_length_seconds (float): Length of the sliding window in seconds.
        hop_length_seconds (float): Hop length for the sliding window in seconds.

    Raises:
        ValueError: If the audio file path is invalid or if the query is empty.
        ValueError: If no audio files are found.
        ValueError: If the audio file extension is not supported.
    """

    # Prepare sample
    audio_path = Path(audio_path)
    if audio_path.is_dir():
        audio_paths = []
        print(f"Searching for audio files in {str(audio_path)} with extensions {', '.join(_AUDIO_FILE_EXTENSIONS)}")
        for ext in _AUDIO_FILE_EXTENSIONS:
            audio_paths.extend(list(audio_path.rglob(f"*{ext}")))

        print(f"Found {len(audio_paths)} audio files in {str(audio_path)}")
    else:
        # check that the extension is valid
        if not any(audio_path.suffix == ext for ext in _AUDIO_FILE_EXTENSIONS):
            raise ValueError(
                f"Invalid audio file extension. Supported extensions are: {', '.join(_AUDIO_FILE_EXTENSIONS)}"
            )
        audio_paths = [audio_path]

    # check that query is not empty
    if not query:
        raise ValueError("Query cannot be empty")
    if not audio_paths:
        raise ValueError("No audio files found. Please check the path or file extensions.")

    # Load model and config
    model, cfg = load_model_and_config(cfg_path)

    # Load audio processor
    processor = NatureLMAudioProcessor(sample_rate=SAMPLE_RATE, max_length_seconds=MAX_LENGTH_SECONDS)

    # Run inference
    results = {"audio_path": [], "output": []}
    for path in audio_paths:
        output = sliding_window_inference(path, query, processor, model, cfg, window_length_seconds, hop_length_seconds)
        results["audio_path"].append(str(path))
        results["output"].append(output)
        print(f"Processed {path}, model output:\n=======\n{output}\n=======\n")

    # Save results as a csv
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_json(output_path, orient="records", lines=True)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        cfg_path=args.cfg_path,
        audio_path=args.audio,
        query=args.query,
        output_path=args.output_path,
        window_length_seconds=args.window_length_seconds,
        hop_length_seconds=args.hop_length_seconds,
    )
