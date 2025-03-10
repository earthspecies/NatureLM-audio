import logging
import os
from functools import lru_cache
from typing import Union

import cloudpathlib
from google.cloud.storage.client import Client

logger = logging.getLogger(__name__)


def is_gcs_path(path: Union[str, os.PathLike]) -> bool:
    return str(path).startswith("gs://")


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
