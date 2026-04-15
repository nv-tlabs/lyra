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

from typing import Dict, Type, TypeVar

from cosmos_predict1.diffusion.training.models.extend_model import ExtendDiffusionModel
from cosmos_predict1.diffusion.training.models.model import DiffusionModel as VideoDiffusionModel
from cosmos_predict1.diffusion.training.utils.layer_control.peft_control_config_parser import LayerControlConfigParser
from cosmos_predict1.diffusion.training.utils.peft.peft import add_lora_layers, setup_lora_requires_grad
from cosmos_predict1.diffusion.utils.customization.customization_manager import CustomizationType
from cosmos_predict1.utils import misc
from cosmos_predict1.utils.lazy_config import instantiate as lazy_instantiate

T = TypeVar("T")


def video_peft_decorator(base_class: Type[T]) -> Type[T]:
    class PEFTVideoDiffusionModel(base_class):
        def __init__(self, config: dict, fsdp_checkpointer=None):
            super().__init__(config)

        @misc.timer("PEFTVideoDiffusionModel: set_up_model")
        def set_up_model(self):
            config = self.config
            peft_control_config_parser = LayerControlConfigParser(config=config.peft_control)
            peft_control_config = peft_control_config_parser.parse()
            self.model = self.build_model()
            if peft_control_config and peft_control_config["customization_type"] == CustomizationType.LORA:
                add_lora_layers(self.model, peft_control_config)
                num_lora_params = setup_lora_requires_grad(self.model)
                if num_lora_params == 0:
                    raise ValueError("No LoRA parameters found. Please check the model configuration.")
            if config.ema.enabled:
                with misc.timer("PEFTDiffusionModel: instantiate ema"):
                    config.ema.model = self.model
                    self.model_ema = lazy_instantiate(config.ema)
                    config.ema.model = None
            else:
                self.model_ema = None

        def state_dict_model(self) -> Dict:
            return {
                "model": self.model.state_dict(),
                "ema": self.model_ema.state_dict() if self.model_ema else None,
            }

    return PEFTVideoDiffusionModel


@video_peft_decorator
class PEFTVideoDiffusionModel(VideoDiffusionModel):
    pass


@video_peft_decorator
class PEFTExtendDiffusionModel(ExtendDiffusionModel):
    pass
