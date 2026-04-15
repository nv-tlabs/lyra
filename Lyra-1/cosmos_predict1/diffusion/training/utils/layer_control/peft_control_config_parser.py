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

import json
import re
from collections import defaultdict
from typing import Union

from loguru import logger
from omegaconf import DictConfig, ListConfig

from cosmos_predict1.diffusion.utils.customization.customization_manager import CustomizationType
from cosmos_predict1.utils.validator import Float, Int, OneOf


class LayerControlConfigParser:
    """
    Parses a config to select layers, blocks, and subblocks to apply LoRA, PEFT, and other finegrained post-training techniques.
    A base model is first loaded then edits (i.e. LoRA, unfreeze, etc.) are applied to the model. Currently, only LoRA is supported for to_q, to_k, to_v, to_out attention layers.
    See: cosmos_predict1/diffusion/training/utils/peft/lora_config.py and LoRA diffusion post-training for an example of how to create and use a LoRA config.
    The input config is a dictionary with the following keys:
    - enabled: whether to apply the PEFT
    - customization_type: default/global type of PEFT to apply (LoRA, unfreeze, etc.)
    - rank: default/global LoRA rank
    - scale: default/global LoRA scale
    - edits: a list of model edits to apply.
        - blocks: a regex to select the blocks to apply the edit to: eg: r'\b(0|1|25|26)\b'
        - block_edit: a list of subblocks to apply the edit to: eg: ["FA[to_q, to_v]", "CA[to_q, to_v]"].
          Subblock names correspond to FA (Full-Attention), CA (Cross-Attention), FL (FinalLayer), and MLP modules as defined in general_dit.py,
          and the layers (i.e to_q, to_k, to_v, etc.) are defined in corresponding modules in attention.py.
        - customization_type: type of PEFT to apply for the edit (LoRA, unfreeze, etc.) - overrides the global customization_type if provided
        - rank: LoRA rank - overrides the global rank for target blocks and subblocks if provided
        - scale: LoRA scale - overrides the global scale for target blocks and subblocks if provided
    """

    SUBBLOCK_PATTERN = r"^(?P<subblock>.+?)\[(?P<parameters>[^\]]+)\]$"  # determines the subblock type (i.e. "FA[...]")
    LAYER_PATTERN = r"^(?P<layer>.+?)(?::(?P<rank>.+?))?(?::(?P<scale>[\d\.]+))?$"  # determines the layer details (i.e. to_q:8:0.6 or to_q)
    FINAL_LAYER_NAME = "final_layer"
    DEFAULT_ALLOWED_TYPES = {  # subblock type to layer types
        "FA": {"to_q", "to_k", "to_v", "to_out", "ada1", "ada2"},
        "CA": {"to_q", "to_k", "to_v", "to_out", "ada1", "ada2"},
        "MLP": {"l1", "l2", "ada1", "ada2"},
    }

    DEFAULT_VALUE_CONSTRAINTS = (
        {  # field to allowed ranges. these ranges are not prescriptive and can be adjusted as needed.
            "blocks": {"min": 0, "max": 27},
            "rank": {"min": 1, "max": 512},
            "scale": {"min": 1e-5, "max": 64},
        }
    )
    ALLOWED_TYPES_FINAL_LAYER = {"FL": {"l1", "ada1", "ada2"}}

    def __init__(self, config: Union[str, dict] = {}, allowed_types: dict = None, value_constraints: dict = None):
        self.config = self._config_to_dict(config)
        self.enabled = str(self.config.get("enabled", "False")).lower() in (
            "true",
            "1",
            "yes",
        )  # if not set, assume disabled
        if self.enabled and not self.config.get("customization_type", ""):
            raise AttributeError("Must specify a top-level customization_type.")
        self.default_customization_type = CustomizationType.from_value(self.config.get("customization_type", ""))
        self.default_rank = self.config.get("rank", None)
        self.default_scale = self.config.get("scale", None)

        self.allowed_types = allowed_types or self.DEFAULT_ALLOWED_TYPES
        self.value_constraints = value_constraints or self.DEFAULT_VALUE_CONSTRAINTS
        logger.info(
            f"Creating layers config with allowed subblock + layer types: \n{self.allowed_types} and value constraints: \n{self.value_constraints}"
        )
        self.allowed_types_final_layer = self.ALLOWED_TYPES_FINAL_LAYER

        self._set_validators()

        self.all_blocks_str = (
            ",".join(
                str(i)
                for i in range(
                    self.value_constraints.get("blocks").get("min"), self.value_constraints.get("blocks").get("max") + 1
                )
            )
            + ","
            + self.FINAL_LAYER_NAME
        )

        self.edits_per_block = defaultdict(lambda: None)

    def _set_validators(self):
        """
        Sets validators for blocks, subblocks, rank, and scale.

        Raises:
            AttributeError: If value constraints are not properly defined.
        """
        self.subblock_validator = OneOf(default="", options=self.allowed_types.keys())
        self.final_layer_validator = OneOf(default="", options=self.allowed_types_final_layer.keys())
        self.rank_validator = None
        self.scale_validator = None
        try:
            self.rank_validator = Int(
                default=0,
                min=self.value_constraints.get("rank").get("min"),
                max=self.value_constraints.get("rank").get("max"),
            )
            self.scale_validator = Float(
                default=0,
                min=self.value_constraints.get("scale").get("min"),
                max=self.value_constraints.get("scale").get("max"),
            )
        except AttributeError:
            raise AttributeError(
                "Value Constraints dictionary must contain 'blocks', 'rank', and 'scale' attributes with 'min' and 'max' attributes for each"
            )

    def _config_to_dict(self, config):
        """
        Convert the given config into a dictionary if provided as a string.

        Args:
            config (Union[str, dict]): The configuration as a JSON string or dictionary.

        Returns:
            dict: The configuration as a dictionary.

        Raises:
            ValueError: If the JSON string is invalid.
            TypeError: If the config is not a string or dictionary.
        """
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided")
        elif not isinstance(config, (dict, DictConfig)):
            raise TypeError(f"Config should be either a JSON string or a dictionary, but got {type(config)}")
        return config

    def _parse_blocks_regex(self, regex):
        """
        Parse the 'blocks' regex and return a set of matching block numbers.
        Allowed block numbers: defined in value_constraints, plus 'final_layer'

        Args:
            regex (str): The regex pattern to match block numbers.

        Returns:
            set: A set of block numbers that match the regex.

        Raises:
            ValueError: If the regex pattern is invalid or matches invalid block numbers.
            Exception: If 'final_layer' is defined with other blocks.
        """
        try:
            block_matches = re.findall(regex, self.all_blocks_str)
            block_numbers = set()
            for match in block_matches:
                match = match.strip()
                if match == "final_layer":
                    block_numbers.add(match)
                else:
                    try:
                        block_numbers.add(int(match))
                    except ValueError:
                        raise ValueError(f"Invalid match found: '{match}' is neither an integer nor 'final_layer'.")
        except re.error as e:
            raise ValueError(f"Invalid regex pattern provided: {regex}. Error: {e}")

        # as final_layer contains a different block type than other blocks, must be defined separately
        if "final_layer" in block_numbers and len(block_numbers) > 1:
            raise Exception(f"Block 'final_layer' must be defined separately, but got: {block_numbers}")

        return block_numbers

    def _parse_subblocks(
        self,
        block_edit: list | ListConfig,
        customization_type: str,
        rank: int,
        scale: float,
        is_final_layer: bool = False,
    ):
        """Generate a dictionary of edits config by subblock.

        Args:
            block_edit (list): List of representing subblocks to apply the edit to (i.e  ["FA[to_q, to_v]", "CA[to_q, to_v]"])
            customization_type (str): The type of PEFT to apply.
            rank (int): The LoRA rank.
            scale (float): The LoRA scale.
            is_final_layer (bool): Indicates if this edit is for the final layer.

        Returns:
            defaultdict: A dictionary of subblock edits configs.

        Raises:
            TypeError: If block_edit is not a list.
            AttributeError: If subblock format is incorrect or layer format is invalid.
            ValueError: If rank and scale values are not provided.
        """
        sb_dict = defaultdict(lambda: None)

        if not isinstance(block_edit, (list, ListConfig)):
            raise TypeError(f"Config 'block_edits' field must be a list, but got {type(block_edit)}")

        if is_final_layer:  # final layer has different allowed layer names
            subblock_validator = self.final_layer_validator
            allowed_types = self.allowed_types_final_layer
        else:
            subblock_validator = self.subblock_validator
            allowed_types = self.allowed_types

        for subblock in block_edit:
            sb_name = None
            params_list = None
            try:
                sb_match = re.match(self.SUBBLOCK_PATTERN, subblock)
                sb_name = subblock_validator.validate(sb_match.group("subblock"))
                params_str = sb_match.group("parameters")
                params_list = params_str.replace(" ", "").split(",")
            except AttributeError:
                raise AttributeError("Incorrect sub-block format: must be <SUBBLOCK_TYPE>[...]")
            layer_validator = OneOf(default="", options=allowed_types.get(sb_name))

            # for each parameter in the subblock config
            layers_dict = defaultdict(lambda: None)
            for param in params_list:
                try:
                    layer_match = re.match(self.LAYER_PATTERN, param)
                    layer_name = layer_validator.validate(layer_match.group("layer"))
                    layer_rank = layer_match.group("rank") or rank or self.default_rank
                    layer_scale = layer_match.group("scale") or scale or self.default_scale
                    if not layer_rank or not layer_scale:
                        raise ValueError(
                            "Rank and scale values must be provided at default, sub-block, or layer level."
                        )
                    layer_rank = self.rank_validator.validate(layer_rank)
                    layer_scale = self.scale_validator.validate(layer_scale)

                    layers_dict[layer_name] = {"activate": True, "lora_rank": layer_rank, "lora_scale": layer_scale}
                    layers_dict["customization_type"] = customization_type or self.default_customization_type
                    sb_dict[sb_name] = dict(layers_dict)
                except AttributeError:
                    raise AttributeError("Layer format must be <layer>:<rank>[:<scale>] (where <scale> is optional)")

        if sb_dict:
            sb_dict["customization_type"] = customization_type or self.default_customization_type
        return sb_dict

    def parse(self):
        """
        Parse the loaded config into a dictionary of edit configs by block number.

        Returns:
            dict: A dictionary of edit configs applied to each block.

        Raises:
            Exception: If more than one edit is specified for a block.
        """
        if not self.enabled:
            return {}

        # for each edit in the config
        for edit in self.config.get("edits", []):
            blocks = self._parse_blocks_regex(edit["blocks"])  # get the blocks affected by edit
            logger.info(f"Applying edits for blocks {blocks}")
            block_edit = edit.get("block_edit", [])
            customization_type = CustomizationType.from_value(edit.get("customization_type", ""))
            rank = edit.get("rank", None)
            scale = edit.get("scale", None)
            is_final_layer = blocks == set([self.FINAL_LAYER_NAME])
            # get subblock config
            sb_dict = self._parse_subblocks(
                block_edit=block_edit,
                customization_type=customization_type,
                rank=rank,
                scale=scale,
                is_final_layer=is_final_layer,
            )

            # for each block in the edit
            for block in blocks:
                if sb_dict:
                    if self.edits_per_block[block]:
                        raise Exception(f"More than one edit specified for block {block}")
                    self.edits_per_block[block] = dict(sb_dict)
        if self.edits_per_block:
            self.edits_per_block["customization_type"] = self.default_customization_type
        return dict(self.edits_per_block)
