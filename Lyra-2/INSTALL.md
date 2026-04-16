# Installation

Tested on Ubuntu 22.04 with CUDA 12.8 and NVIDIA H100 GPUs. Other Linux distributions and CUDA 12.4+ should also work but are not officially verified. Due to the complexity of the dependency stack, version conflicts may arise on different system configurations — if so, please open an issue.

```bash
# 0. Clone repository
git clone --recursive git@github.com:nv-tlabs/lyra.git
cd Lyra-2

# 1. Create conda environment
conda create -n lyra2 python=3.10 pip cmake ninja libgl ffmpeg packaging -c conda-forge -y
conda activate lyra2
CONDA_BACKUP_CXX="" conda install gcc=13.3.0 gxx=13.3.0 eigen zlib -c conda-forge -y

# 2. Install CUDA toolkit inside the conda environment
conda install cuda -c nvidia/label/cuda-12.8.0 -y
export CUDA_HOME=$CONDA_PREFIX

# 3. Install PyTorch
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

# 4. Set build environment variables
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$CPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# 5. Install Python dependencies
pip install --no-deps -r requirements.txt
pip install "git+https://github.com/microsoft/MoGe.git"
pip install --no-build-isolation "transformer_engine[pytorch]"
# Symlink cuda_runtime as cudart for transformer_engine compatibility
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

# 6. Install Flash Attention
MAX_JOBS=16 pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3

# 7. Build vendored CUDA extensions
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'
```

Add the following to your shell profile (e.g. `~/.bashrc`) to persist `LD_LIBRARY_PATH` across sessions:

```bash
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

```bash
# 8. Verify installation
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"
PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference --help
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon --help
```
