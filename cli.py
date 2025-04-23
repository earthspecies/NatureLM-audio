from pathlib import Path

import click
import torch

from beans_zero_inference import main as beans_zero_infer_fn
from inference_web_app import main as app_fn
from NatureLM.infer import main as infer_fn
from train import main as train_fn


def common_options(f):
    f = click.option("--cfg-path", required=True, type=Path, help="Path to configuration file")(f)
    f = click.option(
        "--options",
        default=[],
        multiple=True,
        type=str,
        help="Override fields in the config. A list of key=value pairs",
    )(f)
    return f


@click.command()
@click.option("--cfg-path", required=True, type=Path, help="Path to NatureLM model configuration file")
@click.option("--data-path", default="EarthSpeciesProject/BEANS-Zero", type=str, help="Path to the dataset")
@click.option(
    "--beans-zero-config-path",
    default="beans_zero_dataset_config.json",
    type=Path,
    help="Path to the BEANS config file",
)
@click.option(
    "--output-path", default="beans_zero_eval_output.jsonl", type=Path, help="Path to save the output results"
)
@click.option("--batch-size", default=16, type=int, help="Batch size for inference")
@click.option("--num-workers", default=0, type=int, help="Number of workers for DataLoader")
def beans_zero(cfg_path, data_path, beans_zero_config_path, output_path, batch_size, num_workers):
    """Run inference on the BEANS-Zero dataset."""
    beans_zero_infer_fn(
        cfg_path=cfg_path,
        beans_zero_config_path=beans_zero_config_path,
        data_path=data_path,
        output_path=output_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )


@click.group()
def naturelm():
    pass


@naturelm.command()
@common_options
def train(cfg_path, options):
    train_fn(cfg_path=cfg_path, options=options)


@naturelm.command()
@click.option("--cfg-path", required=True, type=Path, help="Path to NatureLM model configuration file")
@click.option("--audio-path", required=True, type=Path, help="Path to the audio file or directory")
@click.option("--query", required=True, type=str, help="Query for the model")
@click.option("--output-path", default="inference_output.jsonl", type=Path, help="Output path for the results")
@click.option("--window-length-seconds", default=10.0, type=float, help="Length of the sliding window in seconds")
@click.option("--hop-length-seconds", default=10.0, type=float, help="Hop length for the sliding window in seconds")
def infer(cfg_path, audio_path, query, output_path, window_length_seconds, hop_length_seconds):
    infer_fn(
        cfg_path=cfg_path,
        audio_path=audio_path,
        query=query,
        output_path=output_path,
        window_length_seconds=window_length_seconds,
        hop_length_seconds=hop_length_seconds,
    )


@naturelm.command()
@common_options
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
@click.option("--port", default=5001, type=int)
@click.option("--assets-dir", type=Path, default=Path("assets"), help="Path to the directory with static files")
@click.option("--show-errors", type=bool, default=False, help="Show error messages in the web interface")
def inference_app(cfg_path, options, device, port, assets_dir, show_errors):
    app_fn(
        cfg_path=cfg_path,
        options=options,
        device=device,
        port=port,
        assets_dir=assets_dir,
        show_errors=show_errors,
    )


if __name__ == "__main__":
    naturelm()
