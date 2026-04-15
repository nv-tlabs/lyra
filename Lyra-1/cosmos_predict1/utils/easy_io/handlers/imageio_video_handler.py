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

from typing import IO

import numpy as np
import torch

from cosmos_predict1.utils.easy_io.handlers.base import BaseFileHandler

try:
    import imageio
except ImportError:
    imageio = None


class ImageioVideoHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file: IO[bytes], format: str = "mp4", mode: str = "rgb", **kwargs):
        """
        Load video from a file-like object using imageio with specified format and color mode.

        Parameters:
            file (IO[bytes]): A file-like object containing video data.
            format (str): Format of the video file (default 'mp4').
            mode (str): Color mode of the video, 'rgb' or 'gray' (default 'rgb').

        Returns:
            tuple: A tuple containing an array of video frames and metadata about the video.
        """
        file.seek(0)
        video_reader = imageio.get_reader(file, format, **kwargs)

        video_frames = []
        for frame in video_reader:
            if mode == "gray":
                import cv2  # Convert frame to grayscale if mode is gray

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = np.expand_dims(frame, axis=2)  # Keep frame dimensions consistent
            video_frames.append(frame)

        return np.array(video_frames), video_reader.get_meta_data()

    def dump_to_fileobj(
        self,
        obj: np.ndarray | torch.Tensor,
        file: IO[bytes],
        format: str = "mp4",  # pylint: disable=redefined-builtin
        fps: int = 17,
        quality: int = 5,
        **kwargs,
    ):
        """
        Save an array of video frames to a file-like object using imageio.

        Parameters:
            obj (np.ndarray): An array of frames to be saved as video.
            file (IO[bytes]): A file-like object to which the video data will be written.
            format (str): Format of the video file (default 'mp4').
            fps (int): Frames per second of the output video (default 30).

        """
        if isinstance(obj, torch.Tensor):
            assert obj.dtype == torch.uint8
            obj = obj.cpu().numpy()
        h, w = obj.shape[1:-1]
        kwargs = {
            "fps": fps,
            "quality": quality,
            "macro_block_size": 1,
            "ffmpeg_params": ["-s", f"{w}x{h}"],
            "output_params": ["-f", "mp4"],
        }
        imageio.mimsave(file, obj, format, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError
