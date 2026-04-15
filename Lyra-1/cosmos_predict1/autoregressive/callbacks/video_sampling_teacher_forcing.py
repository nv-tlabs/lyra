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

import glob
import math
import os
from typing import Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as torchvision_F
import wandb
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed import get_process_group_ranks

from cosmos_predict1.autoregressive.utils.parallel import (
    broadcast_data_batch_in_tp_cp_group,
    gather_batch_from_cp_ranks,
    get_batch_on_this_cp_rank,
)
from cosmos_predict1.callbacks.every_n import EveryN
from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.model import Model
from cosmos_predict1.utils.trainer import Trainer


def resize_image(image: torch.Tensor, resize_factor=0.5) -> torch.Tensor:
    _, _, h, w = image.shape
    new_h, new_w = int(resize_factor * h), int(resize_factor * w)
    return torchvision_F.resize(image, (new_h, new_w))


class VideoSamplingTeacherForcing(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        video_latent_shape: list = [6, 24, 40],
        num_frames_to_display: int = 4,
        save_folder: Optional[str] = None,
        num_file_to_log: int = 8,
    ):
        r"""
        This callback enables us to perform teacher forcing inference on the training data.
        By teacher forcing, we mean providing ground truth video tokens as inputs, and simply asking the model
        to predict the next tokens. The predicted next tokens are then visualized. This does not perform
        autoregressive sampling.
        We also upload the downsampled video frames to wandb. Downsampling is needed for wandb to work fast.

        Args:
            every_n (int): Call this callback every_n steps
            step_size (int): Number of steps taken for gradient accumulation. Global iteration number is
                iteration // self.step_size
            video_latent_shape (list): Shape of the video latent
            num_frames_to_display (int): Number of frames to subsample for displaying in wandb
            save_folder (str): Name of the local folder to save the video
            num_file_to_log (int): Number of files to upload to wandb
        """
        super().__init__(every_n, step_size)
        self.save_folder = save_folder if save_folder else self.__class__.__name__
        self.video_latent_shape = video_latent_shape
        self.num_frames_to_display = num_frames_to_display
        self.num_file_to_log = num_file_to_log
        self.rank = distributed.get_rank()

    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.save_folder}"
        if self.rank == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Video Teacher-Forcing Callback: local_dir: {self.local_dir}")

    @torch.inference_mode()
    def every_n_impl(
        self,
        trainer: Trainer,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        # Tokenize the data

        broadcast_data_batch_in_tp_cp_group(data_batch)

        input_vid = data_batch[model.tokenizer.tokenizer_config.video_tokenizer.data_key]

        dataset_name = data_batch.get("dataset_name", None)
        if dataset_name is not None and dataset_name.startswith("image"):
            # we disable the callback if the input video is an image batch
            log.info(f"dataset_name is {dataset_name}, skip this callback")
            return

        # get the caption
        captions = data_batch.get("caption", None)

        # get the context embedding and mask
        context = data_batch.get("context", None)
        context_mask = data_batch.get("context_mask", None)
        if context is not None:
            context = misc.to(context, "cuda").detach().clone()
        if context_mask is not None:
            context_mask = misc.to(context_mask, "cuda").detach().clone()
        # get the action
        action = data_batch.get("action", None)
        if action is not None:
            action = misc.to(action, "cuda").detach().clone()

        # Input tokens
        tokens, _ = model.tokenizer.tokenize(data_batch)
        tokens = misc.to(tokens, "cuda").detach().clone()
        skip_save_file = False
        if parallel_state.get_context_parallel_world_size() > 1:
            cp_group = parallel_state.get_context_parallel_group()
            if self.rank != min(get_process_group_ranks(cp_group)):
                skip_save_file = True
            tokens = get_batch_on_this_cp_rank(tokens)
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            # Turn on TP
            tp_group = parallel_state.get_tensor_model_parallel_group()
            if self.rank != min(get_process_group_ranks(tp_group)):
                skip_save_file = True
        tokens_encoded_in_train = output_batch["encode_tokens"].detach()
        percent_token_diff = (tokens != tokens_encoded_in_train).float().mean()
        percent_token_diff = distributed.dist_reduce_tensor(percent_token_diff)

        input_tokens = tokens

        num_tokens_to_generate = np.prod(self.video_latent_shape)

        # Do a forward pass
        logits = model.model.forward(
            tokens,
            input_pos=None,
            context=context,
            context_mask=context_mask,
            action=action,
        )
        if parallel_state.get_context_parallel_world_size() > 1:
            logits = gather_batch_from_cp_ranks(logits)
            input_tokens = gather_batch_from_cp_ranks(input_tokens)

        # Start position for video tokens in the vocabulary
        video_token_start = self.config.model.tokenizer_config.video_tokenizer.tokenizer_offset
        video_vocab_size = self.config.model.tokenizer_config.video_tokenizer.vocab_size

        # Clipping logits only to video tokens. We remove the text vocab predictions.
        # This will ensure that the video tokens only correspond to the video part of the vocabulary.
        logits = logits[:, :, video_token_start : video_token_start + video_vocab_size]

        # Sample with argmax token. This should be good for teacher forcing experiment.
        logits = logits.contiguous()
        generations = torch.argmax(logits, dim=-1)

        # For each video in the batch, subsample frames for display
        batch_size = input_tokens.shape[0]
        out_frames = []
        out_videos_gen = []
        out_videos_rec = []
        out_videos_gt = []
        # log the accuracy of teacher-forcing
        acc = []
        loss_list = []

        for sample_num in range(batch_size):
            # Subsample the generations to the video part.
            # This corresponds to the part from begin of video to end of video.
            bov_token = model.tokenizer.video_special_tokens["<|begin_of_video|>"]
            bov_index = input_tokens[sample_num] == bov_token
            use_special_token = sum(bov_index) != 0
            if use_special_token:
                bov_index = bov_index.nonzero().item()
                # generations: <bov> real_token1 real_token2, ... real_token7680; total 7680
                # gen_video_tokens: real_token1 real_token2, ..., real_token7680; total 7680
                # for vis: real_token1 real_token2, ..., real_token7680; total 7680
                # for accuracy: real_token1 real_token2, ..., real_token7680; total 7680
                gen_video_tokens = generations[sample_num][bov_index : bov_index + num_tokens_to_generate]
                gen_video_tokens_vis = gen_video_tokens
                gen_video_tokens_acc = gen_video_tokens
                logits_loss = logits[sample_num][bov_index : bov_index + num_tokens_to_generate]
            else:
                # generations: real_token1 real_token2, ... real_token7680
                # gen_video_tokens: real_token2 real_token3, ..., real_token7680; total 7679
                # We need different tokens for vis and accuracy compute
                # for acc: real_token2 real_token3, ..., real_token7680; total 7679
                # for vis: pad_token (real_token2, ..., real_token7680); total 1 + 7679
                gen_video_tokens = generations[sample_num][
                    : num_tokens_to_generate - 1
                ]  # remove the last token since there is no gt
                # Since the first token is not predicted, we need to add the gt first token to make sure the shape is correct
                gen_video_tokens_vis = torch.cat([input_tokens[sample_num][0:1], gen_video_tokens])
                gen_video_tokens_acc = gen_video_tokens
                logits_loss = logits[sample_num][: num_tokens_to_generate - 1]

            # Rearrange the video to a spatial tensor
            gen_video_tokens_vis_BTHW = rearrange(
                gen_video_tokens_vis.unsqueeze(0),
                "B (T H W) -> B T H W",
                T=self.video_latent_shape[0],
                H=self.video_latent_shape[1],
                W=self.video_latent_shape[2],
            )

            # for real videos, we need to skip the bov and eov tokens for decoding
            if use_special_token:
                # input_tokens: <bov> real_token1 real_token2 ... <eov> <eov> ...
                # real_video_tokens: real_token1 real_token2 ... real_token7680; total 7680
                # for vis: real_token1 real_token2 ... real_token7680; total 7680
                # for accuracy: real_token1 real_token2 ... real_token7680; total 7680; we include real_token1 since the output prediction also includes it, see gen_video_tokens_acc above
                real_video_tokens = (
                    input_tokens[sample_num][bov_index + 1 : bov_index + num_tokens_to_generate + 1] - video_token_start
                )
                real_video_tokens_vis = real_video_tokens
                real_video_tokens_acc = real_video_tokens
            else:
                # input_tokens: real_token1 real_token2,... real_token7680; total 7680
                # real_video_tokens: real_token1 real_token2,... real_token7680; total 7680
                # for acc: gt start from real_token2, real_token3; total 7679, remove the first token since it is not predicted
                # for vis: gt start from real_token1, real_token2; total 7680
                real_video_tokens = input_tokens[sample_num][:num_tokens_to_generate] - video_token_start
                real_video_tokens_vis = real_video_tokens
                real_video_tokens_acc = real_video_tokens[1:].flatten()

            real_video_tokens_vis_BTHW = rearrange(
                real_video_tokens_vis.unsqueeze(0),
                "B (T H W) -> B T H W",
                T=self.video_latent_shape[0],
                H=self.video_latent_shape[1],
                W=self.video_latent_shape[2],
            )
            # Calculate accuracy
            correct_predictions = (gen_video_tokens_acc == real_video_tokens_acc).float()
            labels = real_video_tokens_acc.clone()

            if model.config.ignore_first_num_tokens > 0:
                labels[: model.config.ignore_first_num_tokens] = model.tokenizer.ignore_index
            select_index = labels != model.tokenizer.ignore_index
            correct_predictions = correct_predictions[select_index]

            loss = torch.nn.functional.cross_entropy(
                logits_loss, labels, ignore_index=model.tokenizer.ignore_index, reduction="none"
            )
            acc.append(correct_predictions.mean() * 100.0)
            loss_list.append(loss.mean())

            # Decode the predicted latents
            if model.tokenizer.tokenizer_config.video_tokenizer.temporal_overlap == 0:
                vid_decoded = model.tokenizer.video_tokenizer.decode(gen_video_tokens_vis_BTHW.cuda())
            else:
                vid_decoded = model.tokenizer.video_tokenizer.decode_with_overlap(
                    gen_video_tokens_vis_BTHW.cuda(),
                    temporal_overlap=model.tokenizer.tokenizer_config.video_tokenizer.temporal_overlap,
                )
            # normalize decoded images from [-1, 1] to [0, 1], and clip value
            vid_decoded = (vid_decoded * 0.5 + 0.5).clamp_(0, 1)
            vid_decoded = vid_decoded[0]

            # Decode the GT latents
            if model.tokenizer.tokenizer_config.video_tokenizer.temporal_overlap == 0:
                vid_rec = model.tokenizer.video_tokenizer.decode(real_video_tokens_vis_BTHW.cuda())
            else:
                vid_rec = model.tokenizer.video_tokenizer.decode_with_overlap(
                    real_video_tokens_vis_BTHW.cuda(),
                    temporal_overlap=model.tokenizer.tokenizer_config.video_tokenizer.temporal_overlap,
                )
            # normalize decoded image from [-1, 1] to [0, 1], and clip value
            vid_rec = (vid_rec * 0.5 + 0.5).clamp_(0, 1)
            vid_rec = vid_rec[0]

            vid_input = input_vid[sample_num]  # [-1, 1], input_vid shape: [B, C, L, H, W]
            vid_input = (vid_input * 0.5 + 0.5).clamp_(0, 1).cuda()  # Convert to [0, 1], [C, L, H, W]

            # Subsample real and generated video frames
            input_video_frames = vid_input.transpose(0, 1)  # [L, C, H, W]
            rec_video_frames = vid_rec.transpose(0, 1)
            gen_video_frames = vid_decoded.transpose(0, 1)
            out_videos_gen.append(gen_video_frames)
            out_videos_rec.append(rec_video_frames)
            out_videos_gt.append(input_video_frames)

            stride = math.ceil(rec_video_frames.shape[0] / self.num_frames_to_display)

            input_video_frames_subsampled = resize_image(input_video_frames[0::stride], resize_factor=0.5)
            input_video_frames_subsampled = torchvision.utils.make_grid(
                input_video_frames_subsampled, nrow=input_video_frames_subsampled.shape[0]
            )

            gt_video_frames_subsampled = resize_image(rec_video_frames[0::stride], resize_factor=0.5)
            gt_video_frames_subsampled = torchvision.utils.make_grid(
                gt_video_frames_subsampled, nrow=gt_video_frames_subsampled.shape[0]
            )
            gen_video_frames_subsampled = resize_image(gen_video_frames[0::stride], resize_factor=0.5)
            gen_video_frames_subsampled = torchvision.utils.make_grid(
                gen_video_frames_subsampled, nrow=gen_video_frames_subsampled.shape[0]
            )

            out_frames.append(input_video_frames_subsampled)
            out_frames.append(gt_video_frames_subsampled)
            out_frames.append(gen_video_frames_subsampled)

        scaled_num_rank_to_log = (
            self.num_file_to_log
            * parallel_state.get_context_parallel_world_size()
            * parallel_state.get_tensor_model_parallel_world_size()
        )
        if self.rank < scaled_num_rank_to_log and not skip_save_file:
            local_path = f"{self.local_dir}/vid_teacher_forcing_iter_{iteration:09d}_{self.rank:04d}.jpg"
            out_image_grid = torchvision.utils.make_grid(out_frames, nrow=1, padding=0, normalize=False)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            torchvision.utils.save_image(out_image_grid, local_path)

        # Log to wandb
        avg_acc = distributed.dist_reduce_tensor(torch.stack(acc).mean()).item()
        avg_loss = distributed.dist_reduce_tensor(torch.stack(loss_list).mean()).item()
        log_info = ""
        if "acc" in output_batch:
            log_info = f"train acc: {(output_batch['acc'].mean().item()):.6f}%"
        if percent_token_diff is not None:
            log_info += f"; percent_token_diff_train_val: {percent_token_diff.item() * 100:.6f}%"
        log.info(
            f"Eval iteration {iteration} teacher-forcing accuracy: {avg_acc:.6f}%, loss: {avg_loss:.4f}; {log_info}"
        )
        if self.rank == 0 and wandb.run:
            local_files = glob.glob(f"{self.local_dir}/vid_teacher_forcing_iter_{iteration:09d}_*.jpg")
            local_files = sorted(local_files)[: self.num_file_to_log]
            if captions is None:
                captions = ["vid_frames_teacher_forcing"] * len(local_files)
            for local_path, caption in zip(local_files, captions):
                wandb.log(
                    {"frames": [wandb.Image(local_path, caption=caption)]},
                    step=iteration,
                )

            wandb.log({"eval/teacher_forcing_acc": avg_acc}, step=iteration)
            wandb.log({"eval/teacher_forcing_loss": avg_loss}, step=iteration)
            if percent_token_diff is not None:
                wandb.log({"eval/percent_token_diff_train_val": percent_token_diff.item() * 100}, step=iteration)
