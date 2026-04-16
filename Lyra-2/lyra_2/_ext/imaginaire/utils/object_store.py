# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import io
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import urlparse

import boto3
import botocore
import numpy as np
import torch
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image

import lyra_2._ext.imaginaire.utils.easy_io.backends.auto_auth as auto
from lyra_2._ext.imaginaire.utils import distributed, log
from lyra_2._ext.imaginaire.utils.easy_io import easy_io

GLOBAL_S3_CONFIG = Config(retries={"max_attempts": 20, "mode": "adaptive"}, connect_timeout=10, read_timeout=60)
Image.MAX_IMAGE_PIXELS = None

if TYPE_CHECKING:
    from lyra_2._ext.imaginaire.config import ObjectStoreConfig


class ObjectStore:
    """This is the interface class for object store, used for interacting with S3-compatible storage.

    Attributes:
        client (botocore.client.S3): Object store client object.
        bucket (str): Object store bucket name.
    """

    def __init__(self, config_object_storage: ObjectStoreConfig):
        with auto.open_auth(config_object_storage.credentials, "r") as file:
            object_storage_config = auto.json_load_auth(file)
            self.client = Boto3Wrapper(
                "s3",
                **object_storage_config,
            )
        self.bucket = config_object_storage.bucket

    def load_object(
        self,
        key: str,
        type: str | None = None,
        load_func: Callable | None = None,
        encoding: str = "UTF-8",
        max_attempts: int = 10,
    ) -> Any:
        """Helper function for loading object from S3.

        Args:
            key (str): The key of the object.
            type (str): Specified for some common data types. If not provided, `load_func` should be specified.
                The predefined types currently supported are:
                - "torch": PyTorch model checkpoints, opened with torch.load().
                - "torch.jit": A JIT-compiled TorchScript model, loaded with torch.jit.load().
                - "image": Image objects, opened with PIL.Image.open().
                - "json": JSON files, opened with json.load().
                - "pickle": Picklable objects, opened with pickle.load().
                - "yaml": YAML files, opened with yaml.safe_load().
                - "text": Pure text files.
                - "numpy": Numpy arrays, opened with np.load().
                - "bytes": Raw bytes.
            load_func (Callable): a custom function for reading the buffer if `type` were not provided.
            encoding (str): Text encoding standard (default: "UTF-8").
            max_attempts (int): Max number of attempts to load the object if there is a failure.

        Returns:
            object (Any): The downloaded object.
        """

        for attempt in range(max_attempts):
            try:
                return self._load_object(
                    key,
                    type=type,
                    load_func=load_func,
                    encoding=encoding,
                )
            except botocore.exceptions.ClientError as e:
                retry_interval = min(0.1 * 2**attempt + random.uniform(0, 1), 30)
                log.exception(
                    f"Failed to load ({self.bucket}) {key}, attempt {attempt}. {e}. Retrying in {retry_interval}s."
                )
                if attempt < max_attempts - 1:
                    time.sleep(retry_interval)
        raise ConnectionError(f"Unable to read ({self.bucket}) {key} after {max_attempts} attempts.")

    def _load_object(
        self, key: str, type: str | None = None, load_func: Callable | None = None, encoding: str = "UTF-8"
    ) -> Any:
        """Helper function for loading object from S3.

        Args:
            key (str): The key of the object.
            type (str): Specified for some common data types. If not provided, `load_func` should be specified.
            load_func (Callable): a custom function for reading the buffer if `type` were not provided.
            encoding (str): Text encoding standard (default: "UTF-8").

        Returns:
            object (Any): The downloaded object.
        """
        assert type is not None or load_func is not None, "Either type or load_func should be specified."
        with io.BytesIO() as buffer:
            self.client.download_fileobj(Bucket=self.bucket, Key=key, Fileobj=buffer)
            buffer.seek(0)
            # Read from buffer for common data types.
            if type == "torch":
                object = torch.load(buffer, map_location=lambda storage, loc: storage, weights_only=False)
            elif type == "torch.jit":
                object = torch.jit.load(buffer)
            elif type == "image":
                object = Image.open(buffer)
                object.load()
            elif type == "json":
                object = json.load(buffer)
            elif type == "pickle":
                object = pickle.load(buffer)
            elif type == "yaml":
                object = yaml.safe_load(buffer)
            elif type == "text":
                object = buffer.read().decode(encoding)
            elif type == "numpy":
                object = np.load(buffer, allow_pickle=True)
            # Read from buffer as raw bytes.
            elif type == "bytes":
                object = buffer.read()
            # Customized load_func should be provided.
            else:
                object = load_func(buffer)
        return object

    def save_object(
        self, object: Any, key: str, type: str | None = None, save_func: Callable | None = None, encoding: str = "UTF-8"
    ) -> None:
        """Helper function for saving object to S3.

        Args:
            object (Any): The object to upload.
            key (str): The key of the object.
            type (str): Specified for some common data types. If not provided, `save_func` should be specified.
                The predefined types currently supported are:
                - "torch": PyTorch model checkpoints, saved with torch.save().
                - "torch.jit": A JIT-compiled TorchScript model, exported with torch.jit.save().
                - "image": Image objects, saved with PIL.Image.save().
                - "json": JSON files, saved with json.dumps().
                - "pickle": Picklable objects, saved with pickle.dump().
                - "yaml": YAML files, saved with yaml.safe_dump().
                - "text": Pure text files.
                - "numpy": Numpy arrays, saved with np.save().
                - "bytes": Raw bytes.
            save_func (Callable): a custom function for writing the buffer if `type` were not provided.
            encoding (str): Text encoding standard (default: "UTF-8").
        """
        assert type is not None or save_func is not None
        with io.BytesIO() as buffer:
            # Write to buffer for common data types.
            if type == "torch":
                torch.save(object, buffer)
            elif type == "torch.jit":
                torch.jit.save(object, buffer)
            elif type == "image":
                type = os.path.basename(key).split(".")[-1]
                object.save(buffer, format=type)
            elif type == "json":
                buffer.write(json.dumps(object).encode(encoding))
            elif type == "pickle":
                pickle.dump(object, buffer)
            elif type == "yaml":
                buffer.write(yaml.safe_dump(object).encode(encoding))
            elif type == "text":
                buffer.write(object.encode(encoding))
            elif type == "numpy":
                np.save(buffer, object)
            # Write to buffer as raw bytes.
            elif type == "bytes":
                buffer.write(bytes(object))
            # Customized save_func should be provided.
            else:
                save_func(object, buffer)
            buffer.seek(0)
            self.client.upload_fileobj(Bucket=self.bucket, Key=key, Fileobj=buffer)

    def object_exists(self, key: str, max_retries: int = 10, retry_delay: float = 2.0) -> bool:
        """
        Check whether an object exists in the storage, with retry logic for transient errors.

        Args:
            key (str): The key of the object.
            max_retries (int): The maximum number of retry attempts in case of errors.
            retry_delay (float): The delay (in seconds) between retry attempts.

        Returns:
            bool: True if the object exists, False if not or if an error persists after retries.
        """
        for attempt in range(max_retries):
            try:
                # Attempt to check if the object exists
                self.client.head_object(Bucket=self.bucket, Key=key)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False  # Object does not exist
                # Log or print the error for troubleshooting
                log.error(f"Attempt {attempt + 1} failed: {e}", rank0_only=False)

                # If this is the last attempt, return False
                if attempt == max_retries - 1:
                    return False

                # Wait for the specified delay before retrying
                time.sleep(retry_delay)
            except Exception as e:
                # Handle other unexpected exceptions
                log.error(f"Unexpected error on attempt {attempt + 1}: {e}", rank0_only=False)

                # If this is the last attempt, return False
                if attempt == max_retries - 1:
                    return False

                # Wait for the specified delay before retrying
                time.sleep(retry_delay)

        # If all retries fail, return False
        return False


class Boto3Wrapper:
    """
    This class serves as a wrapper around boto3.client in order to make boto3.client serializable. It's required to use
    spawn method of creating DataLoader workers, which is in turn required to avoid segfaults when using Triton, e.g.
    for torch.compile or custom kernels.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.client = None

    def __setstate__(self, state):
        self.__dict__ = state

    def __getattr__(self, item):
        is_worker = torch.utils.data.get_worker_info() is not None
        client = (
            boto3.client(*self._args, **self._kwargs, config=GLOBAL_S3_CONFIG) if self.client is None else self.client
        )
        if is_worker:
            self.client = client
        return getattr(client, item)


def sync_s3_dir_to_local(
    s3_dir: str,
    s3_credential_path: str,
    cache_dir: Optional[str] = None,
    rank_sync: bool = True,
    local_rank_sync: bool = False,
) -> str:
    """
    Download an entire directory from S3 to the local cache directory.

    Args:
        s3_dir (str): The AWS S3 directory to download.
        s3_credential_path (str): The path to the AWS S3 credentials file.
        rank_sync (bool, optional): Whether to synchronize download across
            ALL distributed workers using `distributed.barrier()`. Defaults to True.
        cache_dir (str, optional): The cache folder to sync the S3 directory to.
            If None, the environment variable `IMAGINAIRE_CACHE_DIR` (defaulting
            to "~/.cache/imaginaire") will be used.
        local_rank_sync (bool, optional): Whether to synchronize download across
            workers within the same node using a node-level barrier. This is useful
            when the cache directory is not shared across nodes. Defaults to False.
            Note: rank_sync and local_rank_sync cannot both be True.

    Returns:
        local_dir (str): The path to the local directory.
    """
    if local_rank_sync and rank_sync:
        raise ValueError("rank_sync and local_rank_sync cannot be True at the same time.")

    if not s3_dir.startswith("s3://"):
        # If the directory exists locally, return the local path
        assert os.path.exists(s3_dir), f"{s3_dir} is not a S3 path or a local path."
        return s3_dir

    # Get local rank for node-level synchronization
    local_rank = int(os.getenv("LOCAL_RANK", 0)) if local_rank_sync else None

    # Load AWS credentials from the file
    with open(s3_credential_path, "r") as f:
        credentials = json.load(f)

    # Create an S3 client
    s3 = boto3.client(
        "s3",
        **credentials,
    )

    # Parse the S3 URL
    parsed_url = urlparse(s3_dir)
    source_bucket = parsed_url.netloc
    source_prefix = parsed_url.path.lstrip("/")

    # If the local directory is not specified, use the default cache directory
    cache_dir = (
        os.environ.get("IMAGINAIRE_CACHE_DIR", os.path.expanduser("~/.cache/imaginaire"))
        if cache_dir is None
        else cache_dir
    )
    cache_dir = os.path.expanduser(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # List objects in the bucket with the given prefix
    response = s3.list_objects_v2(Bucket=source_bucket, Prefix=source_prefix)
    # Download each matching object
    for obj in response.get("Contents", []):
        if obj["Key"].startswith(source_prefix):
            # Create the full path for the destination file, preserving the directory structure
            rel_path = os.path.relpath(obj["Key"], source_prefix)
            dest_path = os.path.join(cache_dir, source_prefix, rel_path)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Check if the file already exists
            if os.path.exists(dest_path):
                continue
            else:
                log.info(f"Downloading {obj['Key']} to {dest_path}")
                # Download the file
                if rank_sync:
                    # Only rank 0 downloads when using global rank sync
                    if distributed.get_rank() == 0:
                        s3.download_file(source_bucket, obj["Key"], dest_path)
                elif local_rank_sync:
                    # Only local rank 0 (first rank on each node) downloads when using local rank sync
                    if local_rank == 0:
                        s3.download_file(source_bucket, obj["Key"], dest_path)
                else:
                    # No synchronization - every rank downloads
                    s3.download_file(source_bucket, obj["Key"], dest_path)
    # Synchronize after downloads complete
    if rank_sync or local_rank_sync:
        distributed.barrier()

    local_dir = os.path.join(cache_dir, source_prefix)
    return local_dir


def download_from_s3_with_cache(
    s3_path: str,
    s3_credential_path: str,
    cache_fp: Optional[str] = None,
    cache_dir: Optional[str] = None,
    rank_sync: bool = True,
    backend_args: Optional[dict] = None,
    backend_key: Optional[str] = None,
) -> str:
    """download data from S3 with optional caching.

    This function first attempts to load the data from a local cache file. If
    the cache file doesn't exist, it downloads the data from S3 to the cache
    location. Caching is performed in a rank-aware manner
    using `distributed.barrier()` to ensure only one download occurs across
    distributed workers (if `rank_sync` is True).

    Args:
        s3_path (str): The S3 path of the data to load.
        cache_fp (str, optional): The path to the local cache file. If None,
            a filename will be generated based on `s3_path` within `cache_dir`.
        cache_dir (str, optional): The directory to store the cache file. If
            None, the environment variable `IMAGINAIRE_CACHE_DIR` (defaulting
            to "/tmp") will be used.
        rank_sync (bool, optional): Whether to synchronize download across
            distributed workers using `distributed.barrier()`. Defaults to True.
        backend_args (dict, optional): The backend arguments passed to easy_io to construct the backend.
        backend_key (str, optional): The backend key passed to easy_io to registry the backend or retrieve the backend if it is already registered.

    Returns:
        cache_fp (str): The path to the local cache file.

    Raises:
        FileNotFoundError: If the data cannot be found in S3 or the cache.
    """
    if not s3_path.startswith("s3://"):
        # If the file exists locally, return the local path
        assert os.path.exists(s3_path), f"{s3_path} is not a S3 path nor a local path."
        return s3_path

    easy_io.set_s3_backend(
        backend_args={
            "backend": "s3",
            "path_mapping": None,
            "s3_credential_path": s3_credential_path,
        }
    )
    cache_dir = (
        os.environ.get("IMAGINAIRE_CACHE_DIR", os.path.expanduser("~/.cache/imaginaire"))
        if cache_dir is None
        else cache_dir
    )
    cache_dir = os.path.expanduser(cache_dir)
    if cache_fp is None:
        cache_fp = os.path.join(cache_dir, s3_path.replace("s3://", ""))
    if not cache_fp.startswith("/"):
        cache_fp = os.path.join(cache_dir, cache_fp)

    if rank_sync:
        if distributed.get_rank() == 0:
            if os.path.exists(cache_fp):
                # check the size of cache_fp
                if os.path.getsize(cache_fp) < 1:
                    os.remove(cache_fp)
                    log.warning(f"Removed empty cache file {cache_fp}.")

            if not os.path.exists(cache_fp):
                easy_io.copyfile_to_local(
                    s3_path, cache_fp, dst_type="file", backend_args=backend_args, backend_key=backend_key
                )
                log.info(f"Downloaded {s3_path} to {cache_fp}.")
            else:
                log.info(f"The cache file {cache_fp} already exists.")
        distributed.barrier()
    else:
        if os.path.exists(cache_fp):
            # check the size of cache_fp
            if os.path.getsize(cache_fp) < 1:
                os.remove(cache_fp)
                log.warning(f"Removed empty cache file {cache_fp}.")
        if not os.path.exists(cache_fp):
            easy_io.copyfile_to_local(
                s3_path, cache_fp, dst_type="file", backend_args=backend_args, backend_key=backend_key
            )
            log.info(f"Downloaded {s3_path} to {cache_fp}.")
        else:
            log.info(f"The cache file {cache_fp} already exists")
    return cache_fp
