"""Module for training utilities.

This module contains utility functions for training models. For example, saving model checkpoints.
"""

import logging
import os
import tempfile
from typing import Any, Union

import torch
import torch.nn as nn

from NatureLM.storage_utils import GSPath, is_gcs_path

logger = logging.getLogger(__name__)


def maybe_unwrap_dist_model(model: nn.Module, use_distributed: bool) -> nn.Module:
    return model.module if use_distributed else model


def get_state_dict(model, drop_untrained_params: bool = True) -> dict[str, Any]:
    """Get model state dict. Optionally drop untrained parameters to keep only those that require gradient.

    Args:
        model: Model to get state dict from
        drop_untrained_params: Whether to drop untrained parameters

    Returns:
        dict: Model state dict
    """
    if not drop_untrained_params:
        return model.state_dict()

    param_grad_dict = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    return state_dict


def torch_save_to_bucket(save_obj: Any, save_path: Union[str, os.PathLike, GSPath], compress: bool = True) -> None:
    """Save an object directly to GCS bucket without intermediate disk storage.

    Args:
        save_obj: Object to save (usually model state dict or checkpoint)
        save_path: Path to save in GCS bucket (must be gs:// path)
        compress: Whether to use compression. Default: True
    """
    if not is_gcs_path(save_path):
        raise ValueError("save_path must be a GCS path")

    # Convert to GSPath if string
    if isinstance(save_path, (str, os.PathLike)):
        save_path = GSPath(str(save_path))

    # save to a temporary local file and then upload to GCS
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(save_obj, tmp.name, _use_new_zipfile_serialization=compress)
        try:
            save_path.upload_from(tmp.name)
        except Exception as e:
            logger.error(f"Error saving to GCP bucket: {e}")
            raise e


def save_model_checkpoint(
    model: nn.Module,
    save_path: Union[str, os.PathLike, GSPath],
    use_distributed: bool = False,
    drop_untrained_params: bool = False,
    **objects_to_save,
) -> None:
    """Save model checkpoint.

    Args:
        model (nn.Module): Model to save
        output_dir (str): Output directory to save checkpoint
        use_distributed (bool): Whether the model is distributed, if so, unwrap it. Default: False.
        is_best (bool): Whether the model is the best in the training run. Default: False.
        drop_untrained_params (bool): Whether to drop untrained parameters to save. Default: True.
        prefix (str): Prefix to add to the checkpoint file name. Default: "".
        extention (str): Extension to use for the checkpoint file. Default: "pth".
        **objects_to_save: Additional objects to save, e.g. optimizer state dict, etc.
    """
    if not is_gcs_path(save_path) and not os.path.exists(os.path.dirname(save_path)):
        raise FileNotFoundError(f"Directory {os.path.dirname(save_path)} does not exist.")

    model_no_ddp = maybe_unwrap_dist_model(model, use_distributed)
    state_dict = get_state_dict(model_no_ddp, drop_untrained_params)
    save_obj = {
        "model": state_dict,
        **objects_to_save,
    }

    logger.info("Saving checkpoint to {}.".format(save_path))

    if is_gcs_path(save_path):
        torch_save_to_bucket(save_obj, save_path)
    else:
        torch.save(save_obj, save_path)


# Classes that a NatureLM training checkpoint saved before configs were stored as JSON can
# legitimately contain under "config". All are plain value types: none is callable, and
# GSPath pickles without its GCS client or any credentials. Anything outside this set means
# the file is not one of ours, and conversion refuses it.
_LEGACY_CONFIG_GLOBALS = (
    "NatureLM.config.AugmentationsConfig",
    "NatureLM.config.BeatsConfig",
    "NatureLM.config.Config",
    "NatureLM.config.DatasetsConfig",
    "NatureLM.config.GenerateConfig",
    "NatureLM.config.ModelConfig",
    "NatureLM.config.OptimizerConfig",
    "NatureLM.config.RunConfig",
    "NatureLM.storage_utils.GSPath",
    "pathlib.PosixPath",
    "pathlib.PurePosixPath",
    "pathlib.WindowsPath",
    "pathlib.PureWindowsPath",
    "urlparse.ParseResult",
    "urllib.parse.ParseResult",
)

# Pickle records some classes under a module path that no longer exists: ParseResult is a
# namedtuple pickled as "urlparse.ParseResult", but lives in urllib.parse on Python 3.
_LEGACY_GLOBAL_ALIASES = {"urlparse.ParseResult": "urllib.parse.ParseResult"}


def convert_legacy_checkpoint(
    input_path: Union[str, os.PathLike], output_path: Union[str, os.PathLike]
) -> dict[str, Any]:
    """Rewrite a pre-JSON-config checkpoint so it loads under torch.load(weights_only=True).

    Only the "config" entry changes: pickled pydantic objects become plain JSON types.
    Tensors are re-saved untouched.

    The unpickling allowlist is limited to _LEGACY_CONFIG_GLOBALS, so a tampered file cannot
    execute code during conversion -- if it references anything else, we refuse before
    reading it.

    Args:
        input_path: The legacy checkpoint. Local paths and gs:// URIs both work.
        output_path: Where to write the converted checkpoint.

    Returns:
        The converted checkpoint dict.

    Raises:
        ValueError: If the file references globals outside the allowlist.
    """
    import importlib

    from NatureLM.utils import universal_torch_load

    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(os.fspath(input_path)))
    unexpected = sorted(unsafe - set(_LEGACY_CONFIG_GLOBALS))
    if unexpected:
        raise ValueError(
            f"{input_path} references globals that a NatureLM checkpoint should not contain: "
            f"{unexpected}. Refusing to convert; this file may be tampered with."
        )

    # Pair each class with the name the pickle actually records, so torch matches it even
    # when the recorded module path differs from where the class lives today.
    allowed = []
    for name in sorted(unsafe):
        module, _, attr = _LEGACY_GLOBAL_ALIASES.get(name, name).rpartition(".")
        try:
            allowed.append((getattr(importlib.import_module(module), attr), name))
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot resolve {name} referenced by {input_path}: {e}") from e

    # Unpickling bypasses __init__, so GSPath instances have no _handle and their __del__
    # raises at collection. A class-level default makes that a no-op.
    had_handle = hasattr(GSPath, "_handle")
    GSPath._handle = None
    try:
        with torch.serialization.safe_globals(allowed):
            ckpt = universal_torch_load(input_path, cache_mode="use", map_location="cpu")
    finally:
        if not had_handle:
            del GSPath._handle

    config = ckpt.get("config")
    if config is None:
        logger.warning("%s has no 'config' entry; re-saving unchanged", input_path)
    else:
        ckpt["config"] = _config_to_json(config)

    if is_gcs_path(output_path):
        torch_save_to_bucket(ckpt, output_path)
    else:
        torch.save(ckpt, output_path)
    logger.info("Converted %s -> %s", input_path, output_path)
    return ckpt


def _config_to_json(config: Any) -> Any:
    """Recursively reduce pydantic models, Paths and GSPaths to JSON-compatible values."""
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    if isinstance(config, dict):
        return {k: _config_to_json(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [_config_to_json(v) for v in config]
    if isinstance(config, (os.PathLike, GSPath)):
        return str(config)
    return config
