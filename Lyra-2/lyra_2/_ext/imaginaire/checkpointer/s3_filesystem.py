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

import io
import json
import os
import time
from contextlib import contextmanager
from typing import Generator, Union
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystemBase

from lyra_2._ext.imaginaire.utils import log


class S3Stream(io.BytesIO):
    """
    Workaround for PyTorch manually closing the stream before we can upload it to S3. We override the close() as noop
    and instead call our own _true_close() method to close the stream after we are done using it.
    The commit at fault is https://github.com/pytorch/pytorch/commit/9c909bf3bb122db2cce95e2eb7459bbe50dfa15a
    """

    def close(self):
        self.flush()
        # No close

    def _true_close(self):
        super().close()


class S3FileSystem(FileSystemBase):
    """Implementation of FileSystemBase for AWS S3 storage."""

    def __init__(
        self,
        credential_path: str,
        max_attempts: int = 20,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_factor: float = 2.0,
    ) -> None:
        """
        Initialize S3FileSystem with retry configuration.

        Args:
            credential_path: Path to AWS credentials JSON file
            max_attempts: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            backoff_factor: Multiplicative factor for backoff time
        """
        with open(credential_path, "r") as f:
            conf = json.load(f)

        # Configure boto3 with retry settings
        config = Config(
            retries=dict(max_attempts=max_attempts, mode="adaptive"),  # Adaptive mode automatically handles throttling
            connect_timeout=60,
            read_timeout=60,
            request_checksum_calculation="when_required",  # Data integrity check for uploads and downloads
            response_checksum_validation="when_required",  # Data integrity check for uploads and downloads
        )

        self.s3_client = boto3.client("s3", config=config, **conf)
        self.max_attempts = max_attempts
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_factor = backoff_factor

    def _retry_with_backoff(self, operation_func, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry logic.

        Args:
            operation_func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the operation function

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        backoff = self.initial_backoff

        for attempt in range(self.max_attempts):
            try:
                return operation_func(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                log.info(f"S3 Filesystem: Received ClientError: {error_code}", rank0_only=False)

                # Handle specific error cases
                if error_code in ["SlowDown", "ThrottlingException", "RequestLimitExceeded", "InternalError"]:
                    last_exception = e
                    if attempt < self.max_attempts - 1:  # Don't sleep on last attempt
                        current_backoff = min(backoff, self.max_backoff)
                        log.info(f"S3 Filesystem: Retrying in {current_backoff} seconds", rank0_only=False)
                        time.sleep(current_backoff)
                        backoff *= self.backoff_factor
                        continue
                # For other client errors, raise immediately
                raise
            except Exception as e:
                log.info(f"S3 Filesystem: Received Exception: {str(e)}", rank0_only=False)
                last_exception = e
                if attempt < self.max_attempts - 1:
                    current_backoff = min(backoff, self.max_backoff)
                    log.info(f"S3 Filesystem: Retrying in {current_backoff} seconds", rank0_only=False)
                    time.sleep(current_backoff)
                    backoff *= self.backoff_factor
                    continue

        raise last_exception

    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike], mode: str) -> Generator[io.IOBase, None, None]:
        """Create a stream for reading from or writing to S3 with retry logic."""
        path_str = str(path)
        bucket, key = self._parse_s3_uri(path_str)
        log.info(f"S3 Filesystem: Creating stream for {key} in bucket {bucket}", rank0_only=False)

        if mode == "rb":
            stream = io.BytesIO()
            try:

                def download_operation():
                    self.s3_client.download_fileobj(bucket, key, stream)
                    stream.seek(0)

                log.info(f"S3 Filesystem: Downloading {key} from bucket {bucket}", rank0_only=False)
                self._retry_with_backoff(download_operation)
                log.info("S3 Filesystem: Download complete", rank0_only=False)
                yield stream
            finally:
                stream.close()
        elif mode == "wb":
            stream = S3Stream()
            try:
                yield stream

                def upload_operation():
                    stream.seek(0)
                    self.s3_client.upload_fileobj(stream, bucket, key)

                log.info(f"S3 Filesystem: Uploading {key} to bucket {bucket}", rank0_only=False)
                self._retry_with_backoff(upload_operation)
                log.info("S3 Filesystem: Upload complete", rank0_only=False)
            finally:
                stream._true_close()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> Union[str, os.PathLike]:
        """Concatenate S3 path with suffix."""
        path_str = str(path)
        if path_str.endswith("/"):
            return f"{path_str}{suffix}"
        return f"{path_str}/{suffix}"

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """Initialize and validate S3 path."""
        path_str = str(path)
        if not path_str.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {path_str}. Must start with 's3://'")
        return path_str

    def rename(self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]) -> None:
        """Rename (move) an object in S3 with retry logic."""
        src_bucket, src_key = self._parse_s3_uri(str(path))
        dst_bucket, dst_key = self._parse_s3_uri(str(new_path))

        def copy_operation():
            copy_source = {"Bucket": src_bucket, "Key": src_key}
            self.s3_client.copy(copy_source, dst_bucket, dst_key)

        self._retry_with_backoff(copy_operation)

        def delete_operation():
            self.s3_client.delete_object(Bucket=src_bucket, Key=src_key)

        self._retry_with_backoff(delete_operation)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        """
        Create a "directory" in S3.

        Note: S3 doesn't have real directories, but we can create an empty object
        with a trailing slash to simulate a directory.
        """
        # Creating same buckets from different ranks can cause rate limit issues in GCP.
        # In object store, we don't need to create a directory.
        pass

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """Validate if the checkpoint_id is a valid S3 URI."""
        checkpoint_id_str = str(checkpoint_id)
        try:
            if not checkpoint_id_str.startswith("s3://"):
                return False
            parsed = urlparse(checkpoint_id_str)
            return bool(parsed.netloc and parsed.path)  # Must have bucket and key
        except Exception:
            return False

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        """Check if an object exists in S3 with retry logic."""
        bucket, key = self._parse_s3_uri(str(path))
        try:

            def head_operation():
                self.s3_client.head_object(Bucket=bucket, Key=key)

            self._retry_with_backoff(head_operation)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") == "404":
                return False
            raise

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        """Remove a file from S3 with retry logic."""
        bucket, key = self._parse_s3_uri(str(path))

        def delete_operation():
            self.s3_client.delete_object(Bucket=bucket, Key=key)

        self._retry_with_backoff(delete_operation)

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """
        Parse an S3 URI into bucket and key.

        Args:
            uri: S3 URI in the format s3://bucket-name/key

        Returns:
            Tuple of (bucket_name, key)

        Raises:
            ValueError: If the URI is invalid
        """
        uri = uri if isinstance(uri, str) else str(uri)
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")

        parsed = urlparse(uri)
        bucket = parsed.netloc

        # Remove leading slash from key
        key = parsed.path.lstrip("/")

        if not bucket:
            raise ValueError(f"Invalid S3 URI: {uri}. No bucket specified")

        return bucket, key


class S3StorageWriter(FileSystemWriter):
    def __init__(
        self,
        credential_path: str,
        path: str,
        **kwargs,
    ) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (str): The S3 URI to write checkpoints to.
            kwargs (dict): Keyword arguments to pass to the parent :class:`FileSystemWriter`.
        """
        super().__init__(
            path=path,
            sync_files=False,
            **kwargs,
        )
        self.fs = S3FileSystem(credential_path)  # type: ignore
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


class S3StorageReader(FileSystemReader):
    def __init__(self, credential_path: str, path: Union[str, os.PathLike]) -> None:
        """
        Initialize an S3 reader for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (Union[str, os.PathLike]): The S3 path to read checkpoints from.
        """
        super().__init__(path)
        self.fs = S3FileSystem(credential_path)  # type: ignore
        self.path = self.fs.init_path(path)
        self.sync_files = False

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)
