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

import torch
from einops import rearrange
import os
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer
from cosmos_predict1.tokenizer.networks import TokenizerConfigs, TokenizerModels

def load_cosmos_1_decoder(vae_path: str, decoder_cosmos_kwargs):
    tokenizer_cosmos, tokenizer_config = load_cosmos_1_tokenizer(
        checkpoint_path=vae_path,
        load_encoder=False,
        load_decoder=True,
        load_jit=False,
        return_tokenizer_config=True,
        add_tokenizer_kwargs=decoder_cosmos_kwargs,
    )
    decoder = tokenizer_cosmos.decoder
    return decoder, tokenizer_config

def get_tokenizer_config(checkpoint_path: str):
    model_name = os.path.basename(checkpoint_path)
    model_name = model_name.split('Cosmos-Tokenize1-')[1].replace("-", "_")
    tokenizer_config = TokenizerConfigs[model_name].value
    return tokenizer_config

def load_cosmos_1_tokenizer(checkpoint_path: str, load_encoder: bool = True, load_decoder: bool = False, load_jit: bool = True, return_tokenizer_config: bool = False, add_tokenizer_kwargs = None):
    tokenizer_kwargs = {}
    if return_tokenizer_config or not load_jit:
        tokenizer_config = get_tokenizer_config(checkpoint_path)
        tokenizer_name = tokenizer_config["name"]
    else:
        tokenizer_config = None
    if load_encoder:
        tokenizer_kwargs['checkpoint_enc'] = f'{checkpoint_path}/encoder.jit'
    if load_decoder:
        tokenizer_kwargs['checkpoint_dec'] = f'{checkpoint_path}/decoder.jit'
    if not load_jit:
        if add_tokenizer_kwargs:
            for k, v in add_tokenizer_kwargs.items():
                tokenizer_config[k] = v
        tokenizer = TokenizerModels[tokenizer_name].value(**tokenizer_config)
    else:
        tokenizer = CausalVideoTokenizer(**tokenizer_kwargs)
    if return_tokenizer_config:
        return tokenizer, tokenizer_config
    else:
        return tokenizer

def load_cosmos_latent_statistics(vae_path: str, pixel_chunk_duration: int = 121, device: torch.device = 'cpu', weight_dtype: torch.dtype = None):
    tokenizer_config = get_tokenizer_config(vae_path)
    latent_chunk_duration = (pixel_chunk_duration - 1) // tokenizer_config['temporal_compression'] + 1
    latent_mean, latent_std = get_cosmos_diffusion_mean_std(vae_path, weight_dtype, tokenizer_config['latent_channels'], latent_chunk_duration)
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)
    return latent_mean, latent_std

def get_cosmos_diffusion_mean_std(vae_dir: str, dtype: torch.dtype, latent_ch: int, latent_chunk_duration: int):
    latent_mean, latent_std = torch.load(os.path.join(vae_dir, "mean_std.pt"), weights_only=True)
    if dtype is None:
        dtype = latent_mean.dtype
    target_shape = [1, latent_ch, latent_chunk_duration, 1, 1]
    latent_mean = latent_mean.view(latent_ch, -1)
    latent_std = latent_std.view(latent_ch, -1)
    latent_mean = latent_mean.to(dtype).reshape(*target_shape)
    latent_std = latent_std.to(dtype).reshape(*target_shape)
    return latent_mean, latent_std

def denormalize_latents(model_input: torch.Tensor, latent_std: torch.Tensor, latent_mean: torch.Tensor, num_input_multi_views: int = 1, sigma_data: float = 0.5):
    # Add batch dimension
    if len(model_input.shape) == 4:
        model_input = model_input.unsqueeze(0)
        unsqueeze = True
    else:
        unsqueeze = False
    # Use same statistics across views
    model_input = rearrange(model_input, 'b (v t) c h w -> (b v) t c h w', v=num_input_multi_views)
    model_input = model_input / sigma_data
    model_input = model_input * latent_std + latent_mean
    # Convert from generated internal cosmos (B T C H W) to cosmos-predict (B C T H W)
    model_input = model_input.transpose(1, 2)
    # Reshape frames and views again in one dimension
    model_input = rearrange(model_input, '(b v) t c h w -> b (v t) c h w', v=num_input_multi_views)
    # Remove batch dimension
    if unsqueeze:
        model_input = model_input.squeeze(0)
    return model_input

if __name__ == '__main__':
    model_name = 'Cosmos-Tokenize1-CV8x8x8-720p'
    # model_name = 'Cosmos-Tokenize1-CV4x8x8-360p'
    checkpoint_path = f'checkpoints/cosmos_predict1/{model_name}'
    tokenizer = load_cosmos_1_tokenizer(checkpoint_path, load_encoder=True, load_decoder=True)
    input_tensor = torch.rand(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
    input_tensor = input_tensor * 2. - 1.  # Normalize to [-1..1]
    (latent,) = tokenizer.encode(input_tensor)
    torch.testing.assert_close(latent.shape, (1, 16, 3, 64, 64))
    reconstructed_tensor = tokenizer.decode(latent)
    torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)