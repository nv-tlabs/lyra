from typing import Dict, List, Optional

import torch
import torch.nn as nn

from lyra_2._src.modules.clip import CLIPModel
from lyra_2._src.modules.conditioner import AbstractEmbModel


class Wan2pt1CLIPEmbLyra2(AbstractEmbModel):
    """Lyra2-aware CLIP embedder."""

    def __init__(
        self,
        input_key: List[str],
        dropout_rate: float = 0.0,
        num_token: int = 257,
        dtype: str = "bfloat16",
    ):
        super().__init__()
        self.num_token = num_token
        self.model_dim = 1280
        self.clip_model = CLIPModel()

        self._input_key = input_key
        self._output_key = None
        self._dropout_rate = dropout_rate
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

    def random_dropout_input(self, in_tensor=None, dropout_rate=None, key=None):
        return in_tensor

    def forward(
        self,
        image_tensor: Optional[torch.Tensor] = None,
        video_tensor: Optional[torch.Tensor] = None,
        media_latents: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        buffer_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert media_latents is not None, "media_latents is required"
        assert mask is not None, "mask is required"
        with torch.no_grad():
            assert image_tensor is not None, "image_tensor is required"
            context_B_L_D = self.clip_model.visual(image_tensor).to(self.dtype)

        y = torch.concat([mask, media_latents.to(self.dtype)], dim=1)
        out = {"frame_cond_crossattn_emb_B_L_D": context_B_L_D, "y_B_C_T_H_W": y}
        if buffer_latents is not None:
            out["y_buffer_B_C_T_H_W"] = buffer_latents.to(self.dtype)
        return out
