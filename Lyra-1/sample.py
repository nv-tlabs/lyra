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

from typing import Optional, Dict, List
from tqdm import tqdm
import os
import re
import torch
import torch.nn.functional as F
import numpy as np
from accelerate import PartialState
import einops
from omegaconf import OmegaConf
from accelerate.logging import get_logger

from src.models.recon.model_latent_recon import LatentRecon
from src.utils.visu import create_depth_visu, generate_wave_video, save_video
from src.models.data import get_multi_dataloader
from src.models.utils.model import encode_latent_time_vae, encode_plucker_vae
from src.models.utils.render import get_plucker_embedding_and_rays, save_ply, save_ply_orig
from src.models.utils.model import load_vae, encode_multi_view_video, encode_video, decode_multi_view_latents
from src.models.utils.data import write_dict_to_json
from src.models.utils.misc import dtype_map, seed_everything, load_and_merge_configs
from src.models.utils.train import get_most_recent_checkpoint

logger = get_logger(__name__, log_level="INFO")

def load_model(ckpt_path, config, weight_dtype):
    # Load model
    distributed_state = PartialState()
    device = distributed_state.device
    vae = load_vae(config.vae_backbone, config.vae_path)
    transformer = LatentRecon(
        config
    )

    # Load ckpt
    data = torch.load(ckpt_path)
    transformer.load_state_dict(data["module"])

    # Cast model
    transformer.to(device=device, dtype=weight_dtype)
    vae.to(device=device, dtype=weight_dtype)
    transformer.eval()
    vae.eval()
    return transformer, vae, distributed_state

def main(
    config,
    **kwargs
):
    # For dynamic scenes, loop over all target times
    target_index_manual = config.target_index_manual
    if target_index_manual is None and config.target_index_manual_start_idx is not None:
        target_index_manual = list(range(config.target_index_manual_start_idx, config.target_index_manual_start_idx + config.target_index_manual_num_idx, config.target_index_manual_stride))
    if target_index_manual is not None and not isinstance(target_index_manual, int):
        for target_index_manual_manual_i in target_index_manual:
            print(f"Bullet time {target_index_manual_manual_i}")
            config.target_index_manual = target_index_manual_manual_i
            transformer, vae, distributed_state, ckpt_path = main_single(config, **kwargs)
            kwargs['transformer'] = transformer
            kwargs['vae'] = vae
            kwargs['distributed_state'] = distributed_state
            kwargs['ckpt_path'] = ckpt_path
    else:
        main_single(config, **kwargs)

def main_single(
    config,
    seed: int = 0,
    transformer = None,
    vae = None,
    distributed_state = None,
    ckpt_path = None,
):
    weight_dtype = torch.bfloat16
    out_fps = config.out_fps
    g = torch.Generator()
    g.manual_seed(seed)
    seed_everything(seed)
    outdir = config.out_dir_inference

    # Either one config path is given or a list of them to merge them
    if isinstance(config.config_path, str):
        main_config = OmegaConf.load(config.config_path)
    else:
        main_config = load_and_merge_configs(config.config_path)

    # Get latest checkpoint if no checkpoint given (e.g., ckpt_name = 'checkpoint-15000')
    ckpt_name = None
    if ckpt_path is None:
        if config.ckpt_path is None:
            ckpt_model_sub_path = 'pytorch_model/mp_rank_00_model_states.pt'
            ckpts_path = main_config.output_dir
            ckpt_name = config.ckpt_name
            if ckpt_name is None:
                ckpt_name = get_most_recent_checkpoint(ckpts_path)
            ckpt_path = os.path.join(ckpts_path, ckpt_name, ckpt_model_sub_path)
            
        else:
            ckpt_path = config.ckpt_path
    if ckpt_name is None:
        has_ckpt_name = re.search(r"(checkpoint-\d+)", ckpt_path)
        if has_ckpt_name:
            ckpt_name = has_ckpt_name.group(1)
    if ckpt_name is not None:
        outdir = os.path.join(outdir, ckpt_name)
    if os.path.isfile(ckpt_path):
        print(f"Found ckpt at path {ckpt_path}")
    else:
        raise ValueError(f"Could not find ckpt at path {ckpt_path}")
    
    # For dynamic scenes, render all camera viewpoints not only the one from the bullet time
    if config.set_manual_time_idx:
        main_config.set_manual_time_idx = config.set_manual_time_idx
    
    # Set view indices
    if config.static_view_indices_fixed is not None:
        main_config.static_view_indices_fixed = config.static_view_indices_fixed
        outdir = os.path.join(outdir, f"static_view_indices_fixed_{'_'.join(config.static_view_indices_fixed)}")
        main_config.static_view_indices_sampling = 'fixed'
        main_config.num_input_multi_views = len(config.static_view_indices_fixed)
    
    # Subsample the output views
    if config.target_index_subsample is not None:
        main_config.target_index_subsample = config.target_index_subsample
    gaussians_scale_factor = None

    # Define wave visualization parameters
    wave_color_dict = {'wave_color_front': [255, 230, 200], 'wave_color_back': [200, 220, 255], "use_gradient_color": True}
    wave_length = 0.4

    # Export only rgb results for evaluation
    do_eval = config.do_eval
    if do_eval:
        config.save_grid = False
        config.save_gt_input = False
        config.save_gt_depth = False
        config.save_video_input = False
        config.save_rgb_decoding = False
        config.save_gaussians = False
        config.save_gaussians_orig = False
    
    # Generate each sample independently
    main_config.batch_size = 1
    main_config.gs_view_chunk_size = 1

    # We are not using the train data loader
    main_config.num_train_images = 1

    # Set test dataset, otherwise take validation set
    if config.dataset_name is not None:
        main_config.data_mode = [[config.dataset_name, 1]]
        outdir = os.path.join(outdir, config.dataset_name)
    
    # Set bullet time manually
    main_config.target_index_manual = config.target_index_manual
    if config.target_index_manual is not None:
        outdir = os.path.join(outdir, str(config.target_index_manual))
    
    # Set number of test scenes, else take as defined in training config
    if config.num_test_images is not None:
        main_config.num_test_images = config.num_test_images
    
    # Set depth (was only used for supervision)
    main_config.use_depth = config.use_depth

    # Get data loader and model
    train_dataloader, test_dataloader = get_multi_dataloader(main_config)
    if transformer is None and vae is None and distributed_state is None:
        transformer, vae, distributed_state = load_model(ckpt_path, main_config, weight_dtype)
    
    # Set up for grid visualization
    step_test_sum = 0
    step_test_sum_dataset = 0
    test_video_out = []
    test_video_out_rgb = []
    test_video_in = []

    # Output dirs
    outdir_raw = os.path.join(outdir, "raw")
    outdir_meta = os.path.join(outdir, "meta")
    outdir_grid = os.path.join(outdir, "grid")
    outdir_full = os.path.join(outdir, "full_output")
    outdir_3dgs = os.path.join(outdir, "main_gaussians_renderings")
    for d in [outdir, outdir_raw, outdir_meta, outdir_grid, outdir_full, outdir_3dgs]:
        os.makedirs(d, exist_ok=True)
    
    # Loop over test set
    for idx, batch_test in tqdm(enumerate(test_dataloader)):
        
        # Skip based on filter list
        batch_file_name = batch_test['file_name']
        
        # Skip if already generated
        meta_data_sample = {'file_name': batch_file_name}
        meta_data_out_path = os.path.join(outdir_meta, f'sample_{idx}.json')
        if os.path.isfile(meta_data_out_path) and config.skip_existing:
            tqdm.write(f"Skipping {batch_file_name} since {meta_data_out_path} already exists")
            continue
        
        # Check if file exists for eval
        if do_eval:
            eval_file_exists = True
            for view_idx in range(main_config.num_input_multi_views):
                outdir_view_idx = os.path.join(outdir, str(view_idx))
                out_file_name_view = batch_test['file_name']
                assert len(out_file_name_view) == 1, f"More than 1 file_names: {len(out_file_name_view)}"
                out_file_name_view = out_file_name_view[0]
                out_file_path_view = os.path.join(outdir_view_idx, out_file_name_view)
                if not os.path.isfile(f"{out_file_path_view}.mp4"):
                    eval_file_exists = False
                    break
            if eval_file_exists and config.skip_existing:
                tqdm.write(f"Skipping {out_file_name_view} since it already exists")
                continue
        
        # Move to device and cast tensors
        for batch_k, batch_v in batch_test.items():
            if not isinstance(batch_v, torch.Tensor):
                continue
            batch_test[batch_k] = batch_v.to(distributed_state.device)
            # Do rendering with full precision
            if batch_k not in ['intrinsics_input', 'c2ws_input', 'cam_view', 'intrinsics', 'file_name']:
                batch_test[batch_k] = batch_test[batch_k].to(weight_dtype)
        
        # Compute plucker with float64 to match old cpu results
        if main_config.compute_plucker_cuda:
            batch_test['plucker_embedding'], batch_test['rays_os'], batch_test['rays_ds'] = get_plucker_embedding_and_rays(
                batch_test['intrinsics_input'],
                batch_test['c2ws_input'],
                main_config.img_size,
                main_config.patch_size_out_factor,
                batch_test['flip_flag'],
                get_batch_index=False,
                dtype=dtype_map[main_config.compute_plucker_dtype],
                out_dtype=weight_dtype
                )
        
        # Make sure all use the same multi views within one batch
        if 'num_input_multi_views' in batch_test:
            assert (batch_test['num_input_multi_views'][0] == batch_test['num_input_multi_views']).all(), f"Not supporting multi batch size for variable multi-view"
            num_input_multi_views = int(batch_test['num_input_multi_views'][0].item())
            batch_test['num_input_multi_views'] = num_input_multi_views
        
        # Encode video
        if 'rgb_latents' in batch_test:
            model_input = batch_test['rgb_latents'].to(weight_dtype) 
            batch_test['images_input_embed'] = model_input
            video = None
        else:
            video = batch_test['images_input_vae']
            if main_config.use_rgb_decoder:
                model_input = video
            else:
                model_input = encode_multi_view_video(vae, video, num_input_multi_views, main_config.vae_backbone)
            batch_test['images_input_embed'] = model_input
        if main_config.time_embedding_vae:
            batch_test = encode_latent_time_vae(batch_test, lambda x: encode_video(vae, x, main_config.vae_backbone), main_config.img_size)
        if main_config.plucker_embedding_vae:
            batch_test = encode_plucker_vae(batch_test, lambda x: encode_multi_view_video(vae, x, num_input_multi_views, main_config.vae_backbone))
        
        # Reconstruct latents and render from 3DGS
        with torch.no_grad():
            model_output = transformer(batch_test)
        
        # Get RGB and depth from 3DGS
        pred_images = model_output['images_pred'].cpu()
        pred_depths = create_depth_visu(model_output['depths_pred']).cpu()
        if 'depths_output' in batch_test:
            gt_depths = create_depth_visu(batch_test['depths_output'].to(pred_depths.dtype)).cpu()
        else:
            gt_depths = None

        # RGB VAE decoding as reference
        if config.save_rgb_decoding:
            with torch.no_grad():
                reconstructed_latents = decode_multi_view_latents(vae, model_input, num_input_multi_views, main_config.vae_backbone)
            if video is None:
                video = reconstructed_latents
            else:
                video = torch.cat((reconstructed_latents, video), -1)
        
        # Gaussians export just exporting the tensor
        if config.save_gaussians:
            out_dir_gaussians = os.path.join(outdir, 'gaussians')
            os.makedirs(out_dir_gaussians, exist_ok=True)
            path_gaussians = os.path.join(out_dir_gaussians, f'gaussians_{idx}.ply')
            save_ply(model_output['gaussians'], path_gaussians, scale_factor=gaussians_scale_factor)
        
        # Gaussians export following original ply format (used for USDZ with Isaac)
        if config.save_gaussians_orig:
            out_dir_gaussians_orig = os.path.join(outdir, 'gaussians_orig')
            os.makedirs(out_dir_gaussians_orig, exist_ok=True)
            path_gaussians_orig = os.path.join(out_dir_gaussians_orig, f'gaussians_{idx}.ply')
            save_ply_orig(model_output['gaussians'], path_gaussians_orig, scale_factor=gaussians_scale_factor)
        del model_output['gaussians']

        # Wave propagation visualization
        pred_images_views = einops.rearrange(pred_images, 'b (v t) c h w -> v b t c h w', v=num_input_multi_views)
        if not do_eval:
            use_gradient_color = wave_color_dict['use_gradient_color']
            if 'wave_color' in wave_color_dict:
                wave_color = wave_color_dict['wave_color']
                wave_color_front = None
                wave_color_back = None
            else:
                wave_color = None
                wave_color_front = wave_color_dict['wave_color_front']
                wave_color_back = wave_color_dict['wave_color_back']
            pred_images_wave = generate_wave_video(model_output['images_pred'], model_output['depths_pred'], wave_length=wave_length, wave_color=wave_color, use_gradient_color=use_gradient_color, wave_color_front=wave_color_front, wave_color_back=wave_color_back)
            pred_images_rgb = torch.cat((pred_images_wave, pred_images), 1)
            save_video(pred_images_rgb, outdir_3dgs, name=f'rgb_{idx}', fps=out_fps)
            save_video(pred_images_wave, outdir_raw, name=f'rgb_wave_{idx}', fps=out_fps)
            for view_idx, pred_images_view in enumerate(pred_images_views):
                save_video(pred_images_view, outdir_raw, name=f'rgb_{idx}_view_idx_{view_idx}', fps=out_fps)
        
        # Export evaluation rendering with the corresponding filename
        if do_eval:
            for view_idx, pred_images_view in enumerate(pred_images_views):
                outdir_view_idx = os.path.join(outdir, str(view_idx))
                out_file_name_view = batch_test['file_name']
                assert len(out_file_name_view) == 1, f"More than 1 file_names: {len(out_file_name_view)}"
                out_file_name_view = out_file_name_view[0]
                if not os.path.exists(outdir_view_idx):
                    os.makedirs(outdir_view_idx)
                save_video(pred_images_view, outdir_view_idx, name=out_file_name_view, fps=out_fps)
        
        # Add maing 3DGS renderings to grid
        images_grid_list = [pred_images]

        # Add video model RGB reference
        if config.save_gt_input:
            gt_images = batch_test['images_output'].cpu()
            images_grid_list.append(gt_images)
        
        # Input video
        if config.save_video_input and video is not None:
            video_norm = ((video + 1)/2).cpu()
            video_norm = video_norm.float()
            if video_norm.shape == pred_images.shape:
                images_grid_list.append(video_norm)
            else:
                if config.save_grid:
                    test_video_in.append(video_norm)
                save_video(video_norm, outdir_raw, name=f'input_{idx}', fps=out_fps)
        
        # Add images for concatenated visualizations
        images_grid_list.append(pred_depths)
        if config.save_gt_depth and gt_depths is not None:
            images_grid_list.append(gt_depths)
        pred_images_out = torch.cat(images_grid_list, -1)
        step_test_sum += pred_images_out.shape[0]
        if config.save_grid:
            test_video_out.append(pred_images_out)
            test_video_out_rgb.append(pred_images_rgb)
        
        # Write main sample and metadata
        if not do_eval:
            save_video(pred_images_out, outdir_full, name=f'sample_{idx}', fps=out_fps)
            write_dict_to_json(meta_data_sample, meta_data_out_path)

        # Export grid and reset counters
        if step_test_sum >= config.num_grid_samples: 
            if config.save_grid:
                test_video_out = torch.cat(test_video_out, 0)
                save_video(test_video_out, outdir_grid, name=f'sample_grid_{step_test_sum_dataset}', fps=out_fps)
                test_video_out_rgb = torch.cat(test_video_out_rgb, 0)
                save_video(test_video_out_rgb, outdir_grid, name=f'rgb_grid_{step_test_sum_dataset}', fps=out_fps)
                if len(test_video_in) != 0:
                    test_video_in = torch.cat(test_video_in, 0)
                    save_video(test_video_in, outdir_grid, name=f'input_grid_{step_test_sum_dataset}', fps=out_fps)
            step_test_sum = 0
            step_test_sum_dataset += 1
            test_video_out = []
            test_video_out_rgb = []
            test_video_in = []
        tqdm.write(f"Saved batch index {idx} to {outdir}")
        
    tqdm.write(f"Saved all results to {outdir}")
    return transformer, vae, distributed_state, ckpt_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--config_default', type=str, default='configs/inference/default.yaml')
    args, unknown = parser.parse_known_args()
    config = load_and_merge_configs([args.config_default, args.config])
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    main(config)
