from pathlib import Path

import click
import torch

# from beans.run_inference import run_inference as benchmark_fn
from inference_web_app import main as app_fn
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
def beans():
    pass


@click.group()
def naturelm():
    pass


@naturelm.command()
@common_options
def train(cfg_path, options):
    train_fn(cfg_path=cfg_path, options=options)


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
