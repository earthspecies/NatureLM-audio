import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Union

import cloudpathlib
from google.cloud.storage.client import Client

logger = logging.getLogger(__name__)


def is_gcs_path(path: Union[str, os.PathLike]) -> bool:
    return str(path).startswith("gs://")


def is_s3_path(path: Union[str, os.PathLike]) -> bool:
    return str(path).startswith("s3://")


@lru_cache(maxsize=1)
def _get_client():
    return cloudpathlib.GSClient(storage_client=Client())


class GSPath(cloudpathlib.GSPath):
    """
    A wrapper for the GSPath class that provides a default client to the constructor.
    This is necessary due to a bug in cloudpathlib (v0.20.0) which assumes that the
    GOOGLE_APPLICATION_CREDENTIALS environment variable always points to a service
    account. This assumption is incorrect when using Workload Identity Federation, which
    we in our Github Action. Here, we fallback to the actual Google library for a
    default client that handles this correctly.

    For more details, see: https://github.com/drivendataorg/cloudpathlib/issues/390
    """

    def __init__(self, client_path, client=_get_client()):
        super().__init__(client_path, client=client)


def make_bucket_or_dir(path: str | os.PathLike) -> Path | GSPath:
    """Make gcp bucket or directory if it does not exist.

    Args:
        path (str): Path to bucket or directory
    """
    if is_gcs_path(path):
        # TODO replace with cloudpathlib.CloudPath when it is available
        path = GSPath(str(path))
    else:
        path = Path(path)

    if not path.exists():
        logger.info(f"Creating output directory {path}")
        path.mkdir(parents=True, exist_ok=True)

    return path


def upload_directory_to_gcs(
    local_dir: str | os.PathLike,
    dest_folder: str | os.PathLike | GSPath,
    force_overwrite: bool = False,
):
    """
    Upload a local directory to Google Cloud Storage.

    Args:
        local_directory (str): Path to the local directory
        destination_folder (str): Custom destination folder name in GCS.
        force_overwrite (bool): If True, overwrites files in the destination folder.

    Usage:
        upload_directory_to_gcs("local_dir", "gs://bucket-name/destination_folder/")
    """
    if not is_gcs_path(dest_folder):
        raise ValueError("destination_folder must be a GCS path")

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Directory {local_dir} does not exist.")

    # Convert to GSPath if string
    if isinstance(dest_folder, (str, os.PathLike)):
        dest_folder = GSPath(str(dest_folder))

    # Upload directory to GCS
    dest_folder.upload_from(local_dir, force_overwrite_to_cloud=force_overwrite)
