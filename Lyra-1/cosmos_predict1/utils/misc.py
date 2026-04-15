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

import collections
import collections.abc
import functools
import json
import os
import random
import time
from contextlib import ContextDecorator
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, TypeVar
from urllib.parse import urlparse

import boto3
import numpy as np
import termcolor
import torch
from torch import nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._tensor.api import DTensor

from cosmos_predict1.utils import distributed, log
from cosmos_predict1.utils.easy_io import easy_io


def to(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
) -> Any:
    """Recursively cast data into the specified device, dtype, and/or memory_format.

    The input data can be a tensor, a list of tensors, a dict of tensors.
    See the documentation for torch.Tensor.to() for details.

    Args:
        data (Any): Input data.
        device (str | torch.device): GPU device (default: None).
        dtype (torch.dtype): data type (default: None).
        memory_format (torch.memory_format): memory organization format (default: torch.preserve_format).

    Returns:
        data (Any): Data cast to the specified device, dtype, and/or memory_format.
    """
    assert (
        device is not None or dtype is not None or memory_format is not None
    ), "at least one of device, dtype, memory_format should be specified"
    if isinstance(data, torch.Tensor):
        is_cpu = (isinstance(device, str) and device == "cpu") or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        data = data.to(
            device=device,
            dtype=dtype,
            memory_format=memory_format,
            non_blocking=(not is_cpu),
        )
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to(data[key], device=device, dtype=dtype, memory_format=memory_format) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return type(data)([to(elem, device=device, dtype=dtype, memory_format=memory_format) for elem in data])
    else:
        return data


def serialize(data: Any) -> Any:
    """Serialize data by hierarchically traversing through iterables.

    Args:
        data (Any): Input data.

    Returns:
        data (Any): Serialized data.
    """
    if isinstance(data, collections.abc.Mapping):
        return type(data)({key: serialize(data[key]) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return type(data)([serialize(elem) for elem in data])
    else:
        try:
            json.dumps(data)
        except TypeError:
            data = str(data)
        return data


def print_environ_variables(env_vars: list[str]) -> None:
    """Print a specific list of environment variables.

    Args:
        env_vars (list[str]): List of specified environment variables.
    """
    for env_var in env_vars:
        if env_var in os.environ:
            log.info(f"Environment variable {Color.green(env_var)}: {Color.yellow(os.environ[env_var])}")
        else:
            log.warning(f"Environment variable {Color.green(env_var)} not set!")


def set_random_seed(seed: int, by_rank: bool = False) -> None:
    """Set random seed. This includes random, numpy, Pytorch.

    Args:
        seed (int): Random seed.
        by_rank (bool): if true, each GPU will use a different random seed.
    """
    if by_rank:
        seed += distributed.get_rank()
    log.info(f"Using random seed {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets seed on the current CPU & all GPUs


def arch_invariant_rand(
    shape: List[int] | Tuple[int], dtype: torch.dtype, device: str | torch.device, seed: int | None = None
):
    """Produce a GPU-architecture-invariant randomized Torch tensor.

    Args:
        shape (list or tuple of ints): Output tensor shape.
        dtype (torch.dtype): Output tensor type.
        device (torch.device): Device holding the output.
        seed (int): Optional randomization seed.

    Returns:
        tensor (torch.tensor): Randomly-generated tensor.
    """
    # Create a random number generator, optionally seeded
    rng = np.random.RandomState(seed)

    # # Generate random numbers using the generator
    random_array = rng.standard_normal(shape).astype(np.float32)  # Use standard_normal for normal distribution

    # Convert to torch tensor and return
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)


T = TypeVar("T", bound=Callable[..., Any])


class timer(ContextDecorator):  # noqa: N801
    """Simple timer for timing the execution of code.

    It can be used as either a context manager or a function decorator. The timing result will be logged upon exit.

    Example:
        def func_a():
            time.sleep(1)
        with timer("func_a"):
            func_a()

        @timer("func_b)
        def func_b():
            time.sleep(1)
        func_b()
    """

    def __init__(self, context: str, debug: bool = False):
        self.context = context
        self.debug = debug

    def __enter__(self) -> None:
        self.tic = time.time()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        time_spent = time.time() - self.tic
        if self.debug:
            log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")
        else:
            log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")

    def __call__(self, func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN202
            tic = time.time()
            result = func(*args, **kwargs)
            time_spent = time.time() - tic
            if self.debug:
                log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")
            else:
                log.debug(f"Time spent on {self.context}: {time_spent:.4f} seconds")
            return result

        return wrapper  # type: ignore


class TrainingTimer:
    """Timer for timing the execution of code, aggregating over multiple training iterations.

    It is used as a context manager to measure the execution time of code and store the timing results
    for each function. The context managers can be nested.

    Attributes:
        results (dict): A dictionary to store timing results for various code.

    Example:
        timer = Timer()
        for i in range(100):
            with timer("func_a"):
                func_a()
        avg_time = sum(timer.results["func_a"]) / len(timer.results["func_a"])
        print(f"func_a() took {avg_time} seconds.")
    """

    def __init__(self) -> None:
        self.results = dict()
        self.average_results = dict()
        self.start_time = []
        self.func_stack = []
        self.reset()

    def reset(self) -> None:
        self.results = {key: [] for key in self.results}

    def __enter__(self) -> TrainingTimer:
        self.start_time.append(time.time())
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        end_time = time.time()
        result = end_time - self.start_time.pop()
        key = self.func_stack.pop()
        self.results.setdefault(key, [])
        self.results[key].append(result)

    def __call__(self, func_name: str) -> TrainingTimer:
        self.func_stack.append(func_name)
        return self

    def __getattr__(self, func_name: str) -> TrainingTimer:
        return self.__call__(func_name)

    def nested(self, func_name: str) -> TrainingTimer:
        return self.__call__(func_name)

    def compute_average_results(self) -> dict[str, float]:
        results = dict()
        for key, value_list in self.results.items():
            results[key] = sum(value_list) / len(value_list)
        return results


def timeout_handler(timeout_period: float, signum: int, frame: int) -> None:
    # What to do when the process gets stuck. For now, we simply end the process.
    error_message = f"Timeout error: more than {timeout_period} seconds passed since the last iteration."
    raise TimeoutError(error_message)


class Color:
    """A convenience class to colorize strings in the console.

    Example:
        import
        print("This is {Color.red('important')}.")
    """

    @staticmethod
    def red(x: str) -> str:
        return termcolor.colored(str(x), color="red")

    @staticmethod
    def green(x: str) -> str:
        return termcolor.colored(str(x), color="green")

    @staticmethod
    def cyan(x: str) -> str:
        return termcolor.colored(str(x), color="cyan")

    @staticmethod
    def yellow(x: str) -> str:
        return termcolor.colored(str(x), color="yellow")


class BufferCnt:
    """
    Buffer counter which keeps track of the condition when called and returns True when the condition in met "thres"
    amount of times, otherwise returns False.

    Example usage:
        buf = BufferCnt(thres=3)
        for _ in range(5):
            if buf(random.random() > 0.5):
                print("We got lucky 3 times out of 5.")

    Args:
        thres (int): The amount of times the expression needs to be True before returning True.
        reset_over_thres (bool): Whether to reset the buffer after returning True.
    """

    def __init__(self, thres=10, reset_over_thres=False):
        self._cnt = 0
        self.thres = thres
        self.reset_over_thres = reset_over_thres

    def __call__(self, expre, thres=None):
        if expre is True:
            self._cnt += 1
        else:
            self._cnt = 0

        if thres is None:
            thres = self.thres

        if self._cnt >= thres:
            if self.reset_over_thres:
                self.reset()
            return True

        return False

    @property
    def cnt(self):
        return self._cnt

    def reset(self):
        self._cnt = 0


def get_local_tensor_if_DTensor(tensor: torch.Tensor | DTensor) -> torch.tensor:
    if isinstance(tensor, DTensor):
        local = tensor.to_local()
        # As per PyTorch documentation, if the communication is not finished yet, we need to wait for it to finish
        # https://pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.DTensor.to_local
        if isinstance(local, AsyncCollectiveTensor):
            return local.wait()
        else:
            return local
    return tensor


def disabled_train(self: Any, mode: bool = True) -> Any:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def count_params(model: nn.Module, verbose=False) -> int:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def expand_dims_like(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def download_from_s3_with_cache(
    s3_path: str,
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
            None, the environment variable `COSMOS_CACHE_DIR` (defaulting
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
    cache_dir = os.environ.get("TORCH_HOME") if cache_dir is None else cache_dir
    cache_dir = (
        os.environ.get("COSMOS_CACHE_DIR", os.path.expanduser("~/.cache/cosmos")) if cache_dir is None else cache_dir
    )
    cache_dir = os.path.expanduser(cache_dir)
    if cache_fp is None:
        cache_fp = os.path.join(cache_dir, s3_path.replace("s3://", ""))
    if not cache_fp.startswith("/"):
        cache_fp = os.path.join(cache_dir, cache_fp)

    if distributed.get_rank() == 0:
        if os.path.exists(cache_fp):
            # check the size of cache_fp
            if os.path.getsize(cache_fp) < 1:
                os.remove(cache_fp)
                log.warning(f"Removed empty cache file {cache_fp}.")

    if rank_sync:
        if not os.path.exists(cache_fp):
            log.critical(f"Local cache {cache_fp} Not exist! Downloading {s3_path} to {cache_fp}.")
            log.info(f"backend_args: {backend_args}")
            log.info(f"backend_key: {backend_key}")

            easy_io.copyfile_to_local(
                s3_path, cache_fp, dst_type="file", backend_args=backend_args, backend_key=backend_key
            )
            log.info(f"Downloaded {s3_path} to {cache_fp}.")
        else:
            log.info(f"Local cache {cache_fp} already exist! {s3_path} -> {cache_fp}.")

        distributed.barrier()
    else:
        if not os.path.exists(cache_fp):
            easy_io.copyfile_to_local(
                s3_path, cache_fp, dst_type="file", backend_args=backend_args, backend_key=backend_key
            )

            log.info(f"Downloaded {s3_path} to {cache_fp}.")
    return cache_fp


def load_from_s3_with_cache(
    s3_path: str,
    cache_fp: Optional[str] = None,
    cache_dir: Optional[str] = None,
    rank_sync: bool = True,
    backend_args: Optional[dict] = None,
    backend_key: Optional[str] = None,
    easy_io_kwargs: Optional[dict] = None,
) -> Any:
    """Loads data from S3 with optional caching.

    This function first attempts to load the data from a local cache file. If
    the cache file doesn't exist, it downloads the data from S3 to the cache
    location and then loads it. Caching is performed in a rank-aware manner
    using `distributed.barrier()` to ensure only one download occurs across
    distributed workers (if `rank_sync` is True).

    Args:
        s3_path (str): The S3 path of the data to load.
        cache_fp (str, optional): The path to the local cache file. If None,
            a filename will be generated based on `s3_path` within `cache_dir`.
        cache_dir (str, optional): The directory to store the cache file. If
            None, the environment variable `COSMOS_CACHE_DIR` (defaulting
            to "/tmp") will be used.
        rank_sync (bool, optional): Whether to synchronize download across
            distributed workers using `distributed.barrier()`. Defaults to True.
        backend_args (dict, optional): The backend arguments passed to easy_io to construct the backend.
        backend_key (str, optional): The backend key passed to easy_io to registry the backend or retrieve the backend if it is already registered.

    Returns:
        Any: The loaded data from the S3 path or cache file.

    Raises:
        FileNotFoundError: If the data cannot be found in S3 or the cache.
    """
    cache_fp = download_from_s3_with_cache(s3_path, cache_fp, cache_dir, rank_sync, backend_args, backend_key)

    if easy_io_kwargs is None:
        easy_io_kwargs = {}
    return easy_io.load(cache_fp, **easy_io_kwargs)


def sync_s3_dir_to_local(
    s3_dir: str,
    s3_credential_path: str,
    cache_dir: Optional[str] = None,
    rank_sync: bool = True,
) -> str:
    """
    Download an entire directory from S3 to the local cache directory.

    Args:
        s3_dir (str): The AWS S3 directory to download.
        s3_credential_path (str): The path to the AWS S3 credentials file.
        rank_sync (bool, optional): Whether to synchronize download across
            distributed workers using `distributed.barrier()`. Defaults to True.
        cache_dir (str, optional): The cache folder to sync the S3 directory to.
            If None, the environment variable `COSMOS_CACHE_DIR` (defaulting
            to "~/.cache/cosmos") will be used.

    Returns:
        local_dir (str): The path to the local directory.
    """
    if not s3_dir.startswith("s3://"):
        # If the directory exists locally, return the local path
        assert os.path.exists(s3_dir), f"{s3_dir} is not a S3 path or a local path."
        return s3_dir

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
        os.environ.get("COSMOS_CACHE_DIR", os.path.expanduser("~/.cache/cosmos")) if cache_dir is None else cache_dir
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
                if not rank_sync or distributed.get_rank() == 0:
                    s3.download_file(source_bucket, obj["Key"], dest_path)
    if rank_sync:
        distributed.barrier()
    local_dir = os.path.join(cache_dir, source_prefix)
    return local_dir
