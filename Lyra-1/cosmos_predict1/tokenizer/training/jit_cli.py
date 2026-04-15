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

"""A CLI to export an pre-trained tokenizer checkpoint into a torch.ScriptModule.

Usage:
python3 -m cosmos_predict1.tokenizer.training.jit_cli \
    --ckpt_path=checkpoints/Cosmos-0.1-Tokenizer-CV4x8x8/iter_001000000.pt \
    --output_dir=checkpoints/Cosmos-0.1-Tokenizer-CV4x8x8/exported \
    --strict_resume \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
        experiment=CV720_Causal_AE49_4x8x8_cosmos


    will output:
        <output-dir>/iter_001000000_ema.jit
        <output-dir>/iter_001000000_enc.jit
        <output-dir>/iter_001000000_dec.jit

    if --reg is specified, it will export the regular model:
        <output-dir>/iter_001000000_reg.jit
        <output-dir>/iter_001000000_enc.jit
        <output-dir>/iter_001000000_dec.jit
"""

import argparse
import importlib
import os

import torch
from loguru import logger as logging
from torch._dynamo.eval_frame import OptimizedModule as torch_OptimizedModule

from cosmos_predict1.tokenizer.training.checkpointer import TokenizerCheckpointer
from cosmos_predict1.utils import callback, ema
from cosmos_predict1.utils.config import Config
from cosmos_predict1.utils.config_helper import get_config_module, override
from cosmos_predict1.utils.lazy_config import instantiate
from cosmos_predict1.utils.model import Model

parser = argparse.ArgumentParser(description="Export a pre-trained model into a torch.jit.ScriptModule.")
parser.add_argument(
    "--config", type=str, default="cosmos_predict1/tokenizer/training/configs/config.py", help="Path to the config file"
)
parser.add_argument("--ckpt_path", type=str, default=None, help="The full ckpt path.")
parser.add_argument("--credentials", type=str, default="credentials/pdx_vfm_base.secret", help="The credentials file.")
parser.add_argument("--strict_resume", action="store_true", help="Enable strictly loading into every network weight.")
parser.add_argument("--reg", action="store_true", help="Enable regular model export.")
parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory.")

parser.add_argument(
    "opts",
    help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
    """.strip(),
    default=None,
    nargs=argparse.REMAINDER,
)

logging.info("Initialize args, cfg from command line arguments ...")
args = parser.parse_args()
config_module = get_config_module(args.config)
config: Config = importlib.import_module(config_module).make_config()
config = override(config, args.opts)


def _compile_jit_models(model: Model) -> dict[str, torch.ScriptModule]:
    """Returns a TorchScript version of REG or EMA models compiled by PyTorch JIT."""
    assert hasattr(config, "checkpoint") and hasattr(config.checkpoint, "jit")
    config_jit = config.checkpoint.jit
    input_shape = tuple(config_jit.input_shape)
    example_input = torch.randn(input_shape)
    dtype = getattr(torch, config_jit.dtype)
    example_input = example_input.to(config_jit.device).to(dtype)

    # Make sure jit model output consistenly during consecutive calls
    # Check here: https://github.com/pytorch/pytorch/issues/74534
    torch._C._jit_set_texpr_fuser_enabled(False)

    with ema.ema_scope(model, enabled=model.config.ema.enabled and not args.reg):
        _model = model.network.eval()
        if isinstance(_model, torch_OptimizedModule):
            _model = _model._orig_mod
        model_jit = torch.jit.trace(_model, example_input, strict=config_jit.strict)
        encoder_jit = torch.jit.trace(_model.encoder_jit(), example_input, strict=config_jit.strict)
        decoder_example = encoder_jit(example_input)[0]
        decoder_jit = torch.jit.trace(_model.decoder_jit(), decoder_example, strict=config_jit.strict)
    if args.reg:
        return {"reg": model_jit, "enc": encoder_jit, "dec": decoder_jit}
    return {"ema": model_jit, "enc": encoder_jit, "dec": decoder_jit}


def _run_export() -> None:
    """Exports a torch.nn.Module into a torch.jit.ScriptModule."""
    # Check that the config is valid.
    config.validate()
    config.checkpoint.load_path = args.ckpt_path
    config.checkpoint.strict_resume = args.strict_resume
    config.checkpoint.load_training_state = False
    config.job.name = os.path.basename(args.output_dir) if args.output_dir else os.path.basename(args.ckpt_path)

    # Freeze the config.
    config.freeze()  # type: ignore
    callbacks = callback.CallBackGroup(config=config, trainer=None)
    checkpointer = TokenizerCheckpointer(config.checkpoint, config.job, callbacks=callbacks)

    # Create the model.
    logging.info(f"Instantiate model={config.model.config.network.name} ...")
    model = instantiate(config.model)
    model = model.to("cuda", memory_format=config.trainer.memory_format)  # type: ignore
    model.on_train_start(config.trainer.memory_format)

    logging.info(f"loading weights from {config.checkpoint.load_path}...")
    _ = checkpointer.load(model)
    model.eval()
    ckpt_name = config.checkpoint.load_path.split("/")[-1][:-3]

    # Drive the output directory.
    tmp_output_dir = os.path.dirname(config.checkpoint.load_path)
    output_dir = args.output_dir or tmp_output_dir
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Performing JIT compilation ...")
    jit_models = _compile_jit_models(model)
    for name, jit_model in jit_models.items():
        logging.info(f"Outputing torch.jit: {output_dir}/{ckpt_name}_{name}.jit")
        torch.jit.save(jit_model, f"{output_dir}/{ckpt_name}_{name}.jit")


@logging.catch(reraise=True)
def main() -> None:
    _run_export()


if __name__ == "__main__":
    main()
