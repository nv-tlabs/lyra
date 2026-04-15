import argparse
import os
import time

from moge.model.v1 import MoGeModel
import torch
import numpy as np
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.diffusion.inference.gen3c_single_image import (
    create_parser as create_parser_base,
    validate_args as validate_args_base,
    _predict_moge_depth,
    _predict_moge_depth_from_tensor
)
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.distributed import device_with_rank, is_rank0, get_rank
from cosmos_predict1.utils.io import save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer, Cache4D
import torch.nn.functional as F


def create_parser():
    return create_parser_base()


def validate_args(args: argparse.Namespace):
    validate_args_base(args)
    assert args.batch_input_path is None, "Unsupported in persistent mode"
    assert args.prompt is not None, "Prompt is required in persistent mode (but it can be the empty string)"
    assert args.input_image_path is None, "Image should be provided directly by value in persistent mode"
    assert args.trajectory in (None, 'none'), "Trajectory should be provided directly by value in persistent mode, set --trajectory=none"
    assert not args.video_save_name, f"Video saving name will be set automatically for each inference request. Found string: \"{args.video_save_name}\""


def resize_intrinsics(intrinsics: np.ndarray | torch.Tensor,
                      old_size: tuple[int, int], new_size: tuple[int, int],
                      crop_size: tuple[int, int] | None = None) -> np.ndarray | torch.Tensor:
    # intrinsics: (3, 3)
    # old_size: (h1, w1)
    # new_size: (h2, w2)
    if isinstance(intrinsics, np.ndarray):
        intrinsics_copy = np.copy(intrinsics)
    elif isinstance(intrinsics, torch.Tensor):
        intrinsics_copy = intrinsics.clone()
    else:
        raise ValueError(f"Invalid intrinsics type: {type(intrinsics)}")
    intrinsics_copy[:, 0, :] *= new_size[1] / old_size[1]
    intrinsics_copy[:, 1, :] *= new_size[0] / old_size[0]
    if crop_size is not None:
        intrinsics_copy[:, 0, -1] = intrinsics_copy[:, 0, -1] - (new_size[1] - crop_size[1]) / 2
        intrinsics_copy[:, 1, -1] = intrinsics_copy[:, 1, -1] - (new_size[0] - crop_size[0]) / 2
    return intrinsics_copy


class Gen3cPersistentModel():
    """Helper class to run Gen3C image-to-video or video-to-video inference.

    This class loads the models only once and can be reused for multiple inputs.

    This function handles the main video-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts
    """

    @torch.no_grad()
    def __init__(self, args: argparse.Namespace):
        misc.set_random_seed(args.seed)
        validate_args(args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.num_gpus > 1:
            from megatron.core import parallel_state

            from cosmos_predict1.utils import distributed

            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
            process_group = parallel_state.get_context_parallel_group()

        self.frames_per_batch = 121
        self.inference_overlap_frames = 1

        # Initialize video2world generation model pipeline
        pipeline = Gen3cPipeline(
            inference_type="video2world",
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name="Gen3C-Cosmos-7B",
            prompt_upsampler_dir=args.prompt_upsampler_dir,
            enable_prompt_upsampler=not args.disable_prompt_upsampler,
            offload_network=args.offload_diffusion_transformer,
            offload_tokenizer=args.offload_tokenizer,
            offload_text_encoder_model=args.offload_text_encoder_model,
            offload_prompt_upsampler=args.offload_prompt_upsampler,
            offload_guardrail_models=args.offload_guardrail_models,
            disable_guardrail=args.disable_guardrail,
            guidance=args.guidance,
            num_steps=args.num_steps,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_video_frames=self.frames_per_batch,
            seed=args.seed,
        )
        if args.num_gpus > 1:
            pipeline.model.net.enable_context_parallel(process_group)

        self.args = args
        self.frame_buffer_max = pipeline.model.frame_buffer_max
        self.generator = torch.Generator(device=device).manual_seed(args.seed)
        self.sample_n_frames = pipeline.model.chunk_size
        self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        self.pipeline = pipeline
        self.device = device
        self.device_with_rank = device_with_rank(self.device)

        self.cache: Cache3D_Buffer | Cache4D | None = None
        self.model_was_seeded = False
        # User-provided seeding image, after pre-processing.
        # Shape [B, C, T, H, W], type float, range [-1, 1].
        self.seeding_image: torch.Tensor | None = None


    @torch.no_grad()
    def seed_model_from_values(self,
                               images_np: np.ndarray,
                               depths_np: np.ndarray | None,
                               world_to_cameras_np: np.ndarray,
                               focal_lengths_np: np.ndarray,
                               principal_point_rel_np: np.ndarray,
                               resolutions: np.ndarray,
                               masks_np: np.ndarray | None = None):
        import torchvision.transforms.functional as transforms_F

        # Check inputs
        n = images_np.shape[0]
        assert images_np.shape[-1] == 3
        assert world_to_cameras_np.shape == (n, 4, 4)
        assert focal_lengths_np.shape == (n, 2)
        assert principal_point_rel_np.shape == (n, 2)
        assert resolutions.shape == (n, 2)
        assert (depths_np is None) or (depths_np.shape == images_np.shape[:-1])
        assert (masks_np is None) or (masks_np.shape == images_np.shape[:-1])


        if n == 1:
            # TODO: allow user to provide depths, extrinsics and intrinsics
            assert depths_np is None, "Not supported yet: directly providing pre-estimated depth values along with a single image."

            # Note: image is received as 0..1 float, but MoGE expects 0..255 uint8.
            input_image_np = images_np[0, ...] * 255.0
            del images_np

            # Predict depth and initialize 3D cache.
            # Note: even though internally MoGE may use a different resolution, all of the outputs
            # are properly resized & adapted to our desired (self.args.height, self.args.width) resolution,
            # including the intrinsics.
            (
                moge_image_b1chw_float,
                moge_depth_b11hw,
                moge_mask_b11hw,
                moge_initial_w2c_b144,
                moge_intrinsics_b133,
            ) = _predict_moge_depth(
                input_image_np, self.args.height, self.args.width, self.device_with_rank, self.moge_model
            )

            # TODO: MoGE provides camera params, is it okay to just ignore the user-provided ones?
            input_image = moge_image_b1chw_float[:, 0].clone()
            self.cache = Cache3D_Buffer(
                frame_buffer_max=self.frame_buffer_max,
                generator=self.generator,
                noise_aug_strength=self.args.noise_aug_strength,
                input_image=input_image,                     # [B, C, H, W]
                input_depth=moge_depth_b11hw[:, 0],          # [B, 1, H, W]
                # input_mask=moge_mask_b11hw[:, 0],          # [B, 1, H, W]
                input_w2c=moge_initial_w2c_b144[:, 0],       # [B, 4, 4]
                input_intrinsics=moge_intrinsics_b133[:, 0], # [B, 3, 3]
                filter_points_threshold=self.args.filter_points_threshold,
                foreground_masking=self.args.foreground_masking,
            )

            seeding_image = input_image_np.transpose(2, 0, 1)[None, ...] / 128.0 - 1.0
            seeding_image = torch.from_numpy(seeding_image).to(device_with_rank(self.device_with_rank))

            # Return the estimated extrinsics and intrinsics in the same format as the input
            estimated_w2c_b44_np = moge_initial_w2c_b144.cpu().numpy()[:, 0, ...]
            moge_intrinsics_b133_np = moge_intrinsics_b133.cpu().numpy()
            estimated_focal_lengths_b2_np = np.stack([moge_intrinsics_b133_np[:, 0, 0, 0],
                                                    moge_intrinsics_b133_np[:, 0, 1, 1]], axis=1)
            estimated_principal_point_rel_b2_np = moge_intrinsics_b133_np[:, 0, :2, 2]

        else:
            if depths_np is None:
                raise NotImplementedError("Seeding from multiple frames requires providing depth values.")
            if masks_np is None:
                raise NotImplementedError("Seeding from multiple frames requires providing mask values.")

            # RGB: [B, H, W, C] to [B, C, H, W]
            image_bchw_float = torch.from_numpy(images_np.transpose(0, 3, 1, 2).astype(np.float32)).to(self.device_with_rank)
            # Images are received as 0..1 float32, we convert to -1..1 range.
            image_bchw_float = (image_bchw_float * 2.0) - 1.0
            del images_np

            # Depth: [B, H, W] to [B, 1, H, W]
            depth_b1hw = torch.from_numpy(depths_np[:, None, ...].astype(np.float32)).to(self.device_with_rank)
            # Mask: [B, H, W] to [B, 1, H, W]
            mask_b1hw = torch.from_numpy(masks_np[:, None, ...].astype(np.float32)).to(self.device_with_rank)
            # World-to-camera: [B, 4, 4]
            initial_w2c_b44 = torch.from_numpy(world_to_cameras_np).to(self.device_with_rank)
            # Intrinsics: [B, 3, 3]
            intrinsics_b33_np = np.zeros((n, 3, 3), dtype=np.float32)
            intrinsics_b33_np[:, 0, 0] = focal_lengths_np[:, 0]
            intrinsics_b33_np[:, 1, 1] = focal_lengths_np[:, 1]
            intrinsics_b33_np[:, 0, 2] = principal_point_rel_np[:, 0] * self.args.width
            intrinsics_b33_np[:, 1, 2] = principal_point_rel_np[:, 1] * self.args.height
            intrinsics_b33_np[:, 2, 2] = 1.0
            intrinsics_b33 = torch.from_numpy(intrinsics_b33_np).to(self.device_with_rank)

            self.cache = Cache4D(
                input_image=image_bchw_float.clone(), # [B, C, H, W]
                input_depth=depth_b1hw,               # [B, 1, H, W]
                input_mask=mask_b1hw,                 # [B, 1, H, W]
                input_w2c=initial_w2c_b44,            # [B, 4, 4]
                input_intrinsics=intrinsics_b33,      # [B, 3, 3]
                filter_points_threshold=self.args.filter_points_threshold,
                foreground_masking=self.args.foreground_masking,
                input_format=["F", "C", "H", "W"],
            )

            # Return the given extrinsics and intrinsics in the same format as the input
            seeding_image = image_bchw_float
            estimated_w2c_b44_np = world_to_cameras_np
            estimated_focal_lengths_b2_np = focal_lengths_np
            estimated_principal_point_rel_b2_np = principal_point_rel_np

        # Resize seeding image to match the desired resolution.
        if (seeding_image.shape[2] != self.H) or (seeding_image.shape[3] != self.W):
            # TODO: would it be better to crop if aspect ratio is off?
            seeding_image = transforms_F.resize(
                seeding_image,
                size=(self.H, self.W),  # type: ignore
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
        # Switch from [B, C, H, W] to [B, C, T, H, W].
        self.seeding_image = seeding_image[:, :, None, ...]

        working_resolutions_b2_np = np.tile([[self.args.width, self.args.height]], (n, 1))
        return (
            estimated_w2c_b44_np,
            estimated_focal_lengths_b2_np,
            estimated_principal_point_rel_b2_np,
            working_resolutions_b2_np
        )


    @torch.no_grad()
    def inference_on_cameras(self, view_cameras_w2cs: np.ndarray, view_camera_intrinsics: np.ndarray,
                             fps: int | float,
                             overlap_frames:int = 1,
                             return_estimated_depths: bool = False,
                             video_save_quality: int = 5,
                             save_buffer: bool | None = None) -> dict | None:

        # TODO: this is not safe if multiple inference requests are served in parallel.
        # TODO: also, it's not 100% clear whether it is correct to override this request
        #       after initialization of the pipeline.
        self.pipeline.fps = int(fps)
        del fps
        save_buffer = save_buffer if (save_buffer is not None) else self.args.save_buffer

        video_save_name = self.args.video_save_name
        if not video_save_name:
            video_save_name = f"video_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        video_save_path = os.path.join(self.args.video_save_folder, f"{video_save_name}.mp4")
        os.makedirs(self.args.video_save_folder, exist_ok=True)

        cache_is_multiframe = isinstance(self.cache, Cache4D)

        # Note: the inference server already adjusted intrinsics to match our
        # inference resolution (self.W, self.H), so this call is just to make sure
        # that all tensors have the right shape, etc.
        view_cameras_w2cs, view_camera_intrinsics = self.prepare_camera_for_inference(
            view_cameras_w2cs, view_camera_intrinsics,
            old_size=(self.H, self.W), new_size=(self.H, self.W)
        )

        n_frames_total = view_cameras_w2cs.shape[1]
        num_ar_iterations = (n_frames_total - overlap_frames) // (self.sample_n_frames - overlap_frames)
        log.info(f"Generating {n_frames_total} frames will take {num_ar_iterations} auto-regressive iterations")

        # Note: camera trajectory is given by the user, no need to generate it.
        log.info(f"Generating frames 0 - {self.sample_n_frames} (out of {n_frames_total} total)...")
        rendered_warp_images, rendered_warp_masks = self.cache.render_cache(
            view_cameras_w2cs[:, 0:self.sample_n_frames],
            view_camera_intrinsics[:, 0:self.sample_n_frames],
            start_frame_idx=0,
        )

        all_rendered_warps = []
        all_predicted_depth = []
        if save_buffer:
            all_rendered_warps.append(rendered_warp_images.clone().cpu())

        current_prompt = self.args.prompt
        if current_prompt is None and self.args.disable_prompt_upsampler:
            log.critical("Prompt is missing, skipping world generation.")
            return


        # Generate video
        starting_frame = self.seeding_image
        if cache_is_multiframe:
            starting_frame = starting_frame[0].unsqueeze(0)

        generated_output = self.pipeline.generate(
            prompt=current_prompt,
            image_path=starting_frame,
            negative_prompt=self.args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
        )
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            return
        video, _ = generated_output


        def depth_for_frame(frame: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            last_frame_hwc_0_255 = torch.tensor(frame, device=self.device_with_rank)
            pred_image_for_depth_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1) / 255.0 # (C,H,W), range [0,1]

            pred_depth, pred_mask = _predict_moge_depth_from_tensor(
                pred_image_for_depth_chw_0_1, self.moge_model
            )
            return pred_depth, pred_mask, pred_image_for_depth_chw_0_1


        # We predict depth either if we need it (multi-round generation without depth in the cache),
        # or if the user requested it explicitly.
        need_depth_of_latest_frame = return_estimated_depths or (num_ar_iterations > 1 and not cache_is_multiframe)
        if need_depth_of_latest_frame:
            pred_depth, _, pred_image_for_depth_chw_0_1 = depth_for_frame(video[-1])

            if return_estimated_depths:
                # For easier indexing, we include entries even for the frames for which we don't predict
                # depth. Since the results will be transmitted in compressed format, this hopefully
                # shouldn't take up any additional bandwidth.
                depths_batch_0 = np.full((video.shape[0], 1, self.H, self.W), fill_value=np.nan,
                                         dtype=np.float32)
                depths_batch_0[-1, ...] = pred_depth.cpu().numpy()
                all_predicted_depth.append(depths_batch_0)
                del depths_batch_0


        # Autoregressive generation (if needed)
        for num_iter in range(1, num_ar_iterations):
            # Overlap by `overlap_frames` frames
            start_frame_idx = num_iter * (self.sample_n_frames - overlap_frames)
            end_frame_idx = start_frame_idx + self.sample_n_frames
            log.info(f"Generating frames {start_frame_idx} - {end_frame_idx} (out of {n_frames_total} total)...")

            if cache_is_multiframe:
                # Nothing much to do, we assume that depth is alraedy provided and
                # all frames of the seeding video are already in the cache.
                pred_image_for_depth_chw_0_1 = torch.tensor(
                    video[-1], device=self.device_with_rank
                ).permute(2, 0, 1) / 255.0 # (C,H,W), range [0,1]

            else:
                self.cache.update_cache(
                    new_image=pred_image_for_depth_chw_0_1.unsqueeze(0) * 2 - 1, # (B,C,H,W) range [-1,1]
                    new_depth=pred_depth, #  (1,1,H,W)
                    # new_mask=pred_mask,   # (1,1,H,W)
                    new_w2c=view_cameras_w2cs[:, start_frame_idx],
                    new_intrinsics=view_camera_intrinsics[:, start_frame_idx],
                )

            current_segment_w2cs = view_cameras_w2cs[:, start_frame_idx:end_frame_idx]
            current_segment_intrinsics = view_camera_intrinsics[:, start_frame_idx:end_frame_idx]

            cache_start_frame_idx = 0
            if cache_is_multiframe:
                # If requesting more frames than are available in the cache,
                # freeze (hold) on the last batch of frames.
                cache_start_frame_idx = min(
                    start_frame_idx,
                    self.cache.input_frame_count() - (end_frame_idx - start_frame_idx)
                )

            rendered_warp_images, rendered_warp_masks = self.cache.render_cache(
                current_segment_w2cs,
                current_segment_intrinsics,
                start_frame_idx=cache_start_frame_idx,
            )

            if save_buffer:
                all_rendered_warps.append(rendered_warp_images[:, overlap_frames:].clone().cpu())

            pred_image_for_depth_bcthw_minus1_1 = pred_image_for_depth_chw_0_1.unsqueeze(0).unsqueeze(2) * 2 - 1 # (B,C,T,H,W), range [-1,1]
            generated_output = self.pipeline.generate(
                prompt=current_prompt,
                image_path=pred_image_for_depth_bcthw_minus1_1,
                negative_prompt=self.args.negative_prompt,
                rendered_warp_images=rendered_warp_images,
                rendered_warp_masks=rendered_warp_masks,
            )
            video_new, _ = generated_output

            video = np.concatenate([video, video_new[overlap_frames:]], axis=0)

            # Prepare depth prediction for the next AR iteration.
            need_depth_of_latest_frame = return_estimated_depths or ((num_iter < num_ar_iterations - 1) and not cache_is_multiframe)
            if need_depth_of_latest_frame:
                # Either we don't have depth (e.g. single-image seeding), or the user requested
                # depth to be returned explicitly.
                pred_depth, _, pred_image_for_depth_chw_0_1 = depth_for_frame(video_new[-1])
            if return_estimated_depths:
                depths_batch_i = np.full((video_new.shape[0] - overlap_frames, 1, self.H, self.W),
                                         fill_value=np.nan, dtype=np.float32)
                depths_batch_i[-1, ...] = pred_depth.cpu().numpy()
                all_predicted_depth.append(depths_batch_i)
                del depths_batch_i


        if is_rank0():
            # Final video processing
            final_video_to_save = video
            final_width = self.args.width

            if save_buffer and all_rendered_warps:
                squeezed_warps = [t.squeeze(0) for t in all_rendered_warps] # Each is (T_chunk, n_i, C, H, W)

                if squeezed_warps:
                    n_max = max(t.shape[1] for t in squeezed_warps)

                    padded_t_list = []
                    for sq_t in squeezed_warps:
                        # sq_t shape: (T_chunk, n_i, C, H, W)
                        current_n_i = sq_t.shape[1]
                        padding_needed_dim1 = n_max - current_n_i

                        pad_spec = (0,0, # W
                                    0,0, # H
                                    0,0, # C
                                    0,padding_needed_dim1, # n_i
                                    0,0) # T_chunk
                        padded_t = F.pad(sq_t, pad_spec, mode='constant', value=-1.0)
                        padded_t_list.append(padded_t)

                    full_rendered_warp_tensor = torch.cat(padded_t_list, dim=0)

                    T_total, _, C_dim, H_dim, W_dim = full_rendered_warp_tensor.shape
                    buffer_video_TCHnW = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
                    buffer_video_TCHWstacked = buffer_video_TCHnW.contiguous().view(T_total, C_dim, H_dim, n_max * W_dim)
                    buffer_video_TCHWstacked = (buffer_video_TCHWstacked * 0.5 + 0.5) * 255.0
                    buffer_numpy_TCHWstacked = buffer_video_TCHWstacked.cpu().numpy().astype(np.uint8)
                    buffer_numpy_THWC = np.transpose(buffer_numpy_TCHWstacked, (0, 2, 3, 1))

                    final_video_to_save = np.concatenate([buffer_numpy_THWC, final_video_to_save], axis=2)
                    final_width = self.args.width * (1 + n_max)
                    log.info(f"Concatenating video with {n_max} warp buffers. Final video width will be {final_width}")

                else:
                    log.info("No warp buffers to save.")

            # Save video
            save_video(
                video=final_video_to_save,
                fps=self.pipeline.fps,
                H=self.args.height,
                W=final_width,
                video_save_quality=video_save_quality,
                video_save_path=video_save_path,
            )
            log.info(f"Saved video to {video_save_path}")


        if return_estimated_depths:
            predicted_depth = np.concatenate(all_predicted_depth, axis=0)
        else:
            predicted_depth = None


        # Currently `video` is [n_frames, height, width, channels].
        # Return as [1, n_frames, channels, height, width] for consistency with other codebases.
        video = video.transpose(0, 3, 1, 2)[None, ...]
        # Depth is returned as [n_frames, channels, height, width].

        # TODO: handle overlap
        rendered_warp_images_no_overlap = rendered_warp_images
        video_no_overlap = video
        return {
            "rendered_warp_images": rendered_warp_images,
            "video": video,
            "rendered_warp_images_no_overlap": rendered_warp_images_no_overlap,
            "video_no_overlap": video_no_overlap,
            "predicted_depth": predicted_depth,
            "video_save_path": video_save_path,
        }

    # --------------------

    def prepare_camera_for_inference(self, view_cameras: np.ndarray, view_camera_intrinsics: np.ndarray,
                                     old_size: tuple[int, int], new_size: tuple[int, int]):
        """Old and new sizes should be given as (height, width)."""
        if isinstance(view_cameras, np.ndarray):
            view_cameras = torch.from_numpy(view_cameras).float().contiguous()
        if view_cameras.ndim == 3:
            view_cameras = view_cameras.unsqueeze(dim=0)

        if isinstance(view_camera_intrinsics, np.ndarray):
            view_camera_intrinsics = torch.from_numpy(view_camera_intrinsics).float().contiguous()

        view_camera_intrinsics = resize_intrinsics(view_camera_intrinsics, old_size, new_size)
        view_camera_intrinsics = view_camera_intrinsics.unsqueeze(dim=0)
        assert view_camera_intrinsics.ndim == 4

        return view_cameras.to(device_with_rank(self.device_with_rank)), \
               view_camera_intrinsics.to(device_with_rank(self.device_with_rank))


    def get_cache_input_depths(self) -> torch.Tensor | None:
        if self.cache is None:
            return None
        return self.cache.input_depth

    @property
    def W(self) -> int:
        return self.args.width

    @property
    def H(self) -> int:
        return self.args.height


    def clear_cache(self) -> None:
        self.cache = None
        self.model_was_seeded = False


    def cleanup(self) -> None:
        if self.args.num_gpus > 1:
            rank = get_rank()
            log.info(f"Model cleanup: destroying model parallel group on rank={rank}.",
                     rank0_only=False)
            from megatron.core import parallel_state
            parallel_state.destroy_model_parallel()

            import torch.distributed as dist
            dist.destroy_process_group()

            log.info(f"Destroyed model parallel group on rank={rank}.", rank0_only=False)
        else:
            log.info("Model cleanup: nothing to do (no parallelism).", rank0_only=False)
