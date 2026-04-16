from dataclasses import dataclass
from typing import Dict, Optional

import torch
from hydra.core.config_store import ConfigStore

from lyra_2._ext.imaginaire.lazy_config import LazyCall as L
from lyra_2._ext.imaginaire.lazy_config import LazyDict
from lyra_2._src.models.lyra2_model import WAN2PT1_I2V_COND_LATENT_KEY
from lyra_2._src.networks.clip_lyra2 import Wan2pt1CLIPEmbLyra2
from lyra_2._src.modules.conditioner import (
    BaseCondition,
    GeneralConditioner,
    ReMapkey,
    T2VCondition,
    TextAttrEmptyStringDrop,
    broadcast_condition,
)
from lyra_2._src.utils.context_parallel import broadcast


@dataclass(frozen=True)
class Img2VidWan2pt1ConditionLyra2(T2VCondition):
    frame_cond_crossattn_emb_B_L_D: Optional[torch.Tensor] = None
    y_B_C_T_H_W: Optional[torch.Tensor] = None
    y_buffer_B_C_T_H_W: Optional[torch.Tensor] = None

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> BaseCondition:
        if self.is_broadcasted:
            return self
        kwargs = self.to_dict(skip_underscore=False)
        y = kwargs.pop("y_B_C_T_H_W")
        y_buffer = kwargs.pop("y_buffer_B_C_T_H_W")
        new_cond = T2VCondition.broadcast(type(self)(**kwargs), process_group)
        kwargs = new_cond.to_dict(skip_underscore=False)
        kwargs["y_B_C_T_H_W"] = broadcast(y, process_group)
        kwargs["y_buffer_B_C_T_H_W"] = broadcast(y_buffer, process_group) if y_buffer is not None else None
        return type(self)(**kwargs)


class Img2VidWan2pt1ConditionerLyra2(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Img2VidWan2pt1ConditionLyra2:
        output = super()._forward(batch, override_dropout_rate)
        return Img2VidWan2pt1ConditionLyra2(**output)


Lyra2ConditionerConfig: LazyDict = L(Img2VidWan2pt1ConditionerLyra2)(
    text=L(TextAttrEmptyStringDrop)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    wanclip=L(Wan2pt1CLIPEmbLyra2)(
        input_key=["last_hist_frame", "video", WAN2PT1_I2V_COND_LATENT_KEY, "cond_latent_mask", "cond_latent_buffer"],
        dropout_rate=0.0,
        dtype="bfloat16",
    ),
)


def lyra_register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="lyra2_conditioner",
        node=Lyra2ConditionerConfig,
    )
