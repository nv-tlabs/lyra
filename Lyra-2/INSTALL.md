# Lyra-2 Installation Runbook

This document records the RunPod installation that was verified for Lyra-2, including the fixes that were needed during the live setup. It is intended to be usable as a Jira runbook for another teammate.

Verified setup:

- Provider: RunPod
- OS image: Ubuntu 24.04
- GPU: NVIDIA H200 NVL, 143 GB VRAM
- NVIDIA driver: 580.126.16
- Python: 3.10.20
- CUDA toolkit in conda env: 12.8
- PyTorch: 2.7.1+cu128
- Repository path on remote: `/workspace/lyra`
- Lyra-2 path on remote: `/workspace/lyra/Lyra-2`
- Conda env path used in the verified run: `/root/conda-envs/lyra2`
- Checkpoints path: `/workspace/lyra/Lyra-2/checkpoints`

The original upstream install was tested on Ubuntu 22.04, CUDA 12.8, and H100 GPUs. The steps below are the H200/RunPod version we actually used.

## Stability Status

This runbook is verified as a working installation path for one active RunPod session. It is not automatically persistent across every RunPod Stop/Recreate event.

Observed after a pod stop/recreate:

- `/workspace` may persist if the pod is attached to the same persistent RunPod volume.
- `/root` can be reset. If that happens, `/root/conda-envs/lyra2`, `/root/conda-pkgs`, `/root/.cache`, and `/root/lyra2_recon_download` are gone.
- If `/root` is reset, the conda environment must be rebuilt and any checkpoint symlink that points into `/root` must be repaired.

For a teammate following this document, treat the `/root` layout below as the fast, verified live-session setup. For a stop/recreate-safe setup, either keep the pod alive until the work is done, create a reusable image/snapshot after installation, or use a RunPod configuration where the environment is stored on persistent storage and verify it with the health checks in section 18.

## 0. Connect to RunPod

Use the SSH command from the RunPod UI. Example shape:

```bash
ssh root@<RUNPOD_IP> -p <RUNPOD_PORT> -i ~/.ssh/id_ed25519
```

Check the GPU:

```bash
nvidia-smi
ls -l /dev/nvidia* 2>/dev/null || true
```

Expected for the verified pod:

```text
NVIDIA H200 NVL
Driver Version: 580.126.16
Memory: 143771 MiB
/dev/nvidia0
/dev/nvidiactl
/dev/nvidia-uvm
```

If `nvidia-smi` is missing, `/dev/nvidia*` is empty, or PyTorch reports `cuda available: False`, do not continue with Lyra installation or inference. The pod is not running with the NVIDIA GPU runtime mounted, or you are connected to the wrong pod/port. This cannot be fixed from inside the container; recreate/start a GPU pod and verify `nvidia-smi` before spending time on dependencies.

## 1. Important RunPod Storage Rule

Do not put conda package extraction caches under `/workspace` on this pod.

During the live setup, `mamba package extract` processes became stuck in kernel `D` state when package extraction used `/workspace/conda-pkgs` on the network filesystem. Those processes could not be killed until the pod/host cleared them.

The verified live-session layout was:

- Keep repo and model checkpoints in `/workspace`
- Keep conda envs and conda package cache in `/root`

```bash
export CONDA_ENVS_PATH=/root/conda-envs
export CONDA_PKGS_DIRS=/root/conda-pkgs
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" /root/tmp
export TMPDIR=/root/tmp
```

Important persistence caveat: `/root` can be ephemeral on RunPod. This layout avoids the `/workspace` package extraction hang, but it also means the conda environment can disappear after a pod Stop/Recreate. If that happens, rerun sections 4 through 12 before trying inference again.

If the pod must survive Stop/Recreate without rebuilding, use one of these approaches before starting the 3-hour install:

- Use a RunPod image/snapshot/template that preserves the completed `/root/conda-envs/lyra2` environment.
- Attach persistent storage and install the environment there, but keep `CONDA_PKGS_DIRS` and `TMPDIR` on local disk during package extraction. This was not the path verified in the live setup and should be health-checked carefully.
- Keep `/root` ephemeral and accept that the env is disposable; this is reproducible but not restart-persistent.

## 2. Install Miniforge

The RunPod image used during setup did not have conda available, so Miniforge was installed manually.

```bash
cd /workspace
curl -L -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
source /workspace/miniforge3/etc/profile.d/conda.sh
conda config --set auto_activate_base false
conda --version
mamba --version
```

Expected result from the verified run:

```text
conda 26.1.1
```

## 3. Clone Lyra

The SSH clone in the upstream docs can fail if the pod does not have GitHub SSH auth configured. The verified setup used HTTPS.

```bash
cd /workspace
git clone --recursive https://github.com/nv-tlabs/lyra.git
cd /workspace/lyra
git rev-parse --short HEAD
git submodule status
```

Verified repo state:

```text
repo HEAD: 52e5079
Lyra-2/lyra_2/_src/inference/depth_anything_3: 1ed6cb8...
Lyra-2/lyra_2/_src/inference/vipe: b7cac64...
```

Then enter Lyra-2:

```bash
cd /workspace/lyra/Lyra-2
```

## 4. Create the Conda Environment

Pin CUDA to 12.8. During the live setup, an unpinned CUDA install tried to pull a newer CUDA stack, so the env was recreated and CUDA was pinned.

Do not use `set -u` around `conda activate`; one activation script references `SYS_SYSROOT` and fails under `set -u`.

```bash
export CONDA_ENVS_PATH=/root/conda-envs
export CONDA_PKGS_DIRS=/root/conda-pkgs
export TMPDIR=/root/tmp
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$TMPDIR"

source /workspace/miniforge3/etc/profile.d/conda.sh

mamba create -p /root/conda-envs/lyra2 \
  python=3.10 pip cmake ninja libgl ffmpeg packaging \
  -c conda-forge -y

conda activate /root/conda-envs/lyra2

CONDA_BACKUP_CXX="" mamba install \
  gcc=13.3.0 gxx=13.3.0 eigen zlib \
  -c conda-forge -y

mamba install "cuda=12.8" \
  -c nvidia/label/cuda-12.8.0 \
  --override-channels -y
```

Verify:

```bash
python --version
nvcc --version
conda list | egrep '^(cuda|cuda-version|cuda-nvcc|gcc|gxx)\s'
```

Expected:

```text
Python 3.10.20
Cuda compilation tools, release 12.8, V12.8.93
```

## 5. Set Build Environment Variables

Run this in every shell before building extensions or running inference.

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /root/conda-envs/lyra2

export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
SITE=$CONDA_PREFIX/lib/python3.10/site-packages

export CPATH="$CUDA_HOME/include:$SITE/nvidia/nvtx/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPATH:-}"
export C_INCLUDE_PATH="$SITE/nvidia/nvtx/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$SITE/nvidia/nvtx/include:${CPLUS_INCLUDE_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS=16
```

After PyTorch is installed, also create the NVTX compatibility symlinks used by Transformer Engine:

```bash
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
ln -sfn "$SITE/nvidia/nvtx/include/nvtx3" "$CONDA_PREFIX/include/nvtx3"
ln -sfn "$SITE/nvidia/nvtx/include/nvToolsExt.h" "$CONDA_PREFIX/include/nvToolsExt.h"
```

Why this is needed: the first Transformer Engine build failed with:

```text
fatal error: nvtx3/nvToolsExt.h: No such file or directory
```

The header existed under `site-packages/nvidia/nvtx/include`, but was not in the compiler include path.

## 6. Install PyTorch

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /root/conda-envs/lyra2

python -m pip install --upgrade pip setuptools wheel
pip install torch==2.7.1 torchvision==0.22.1 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Expected:

```text
torch 2.7.1+cu128
cuda True
gpu NVIDIA H200 NVL
```

## 7. Install Base Python Dependencies

```bash
cd /workspace/lyra/Lyra-2

pip install --no-deps -r requirements.txt
pip install "git+https://github.com/microsoft/MoGe.git"
```

The `--no-deps` install can leave some optional test/dev packages missing. This was acceptable for inference. The runtime fixes that were required are listed below.

## 8. Install Transformer Engine

Make sure the build environment variables and NVTX symlinks from step 5 are active.

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /root/conda-envs/lyra2

export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
ln -sfn "$SITE/nvidia/nvtx/include/nvtx3" "$CONDA_PREFIX/include/nvtx3"
ln -sfn "$SITE/nvidia/nvtx/include/nvToolsExt.h" "$CONDA_PREFIX/include/nvToolsExt.h"

export CPATH="$CUDA_HOME/include:$SITE/nvidia/nvtx/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPATH:-}"
export C_INCLUDE_PATH="$SITE/nvidia/nvtx/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$SITE/nvidia/nvtx/include:${CPLUS_INCLUDE_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS=16

pip install --no-build-isolation "transformer_engine[pytorch]"
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"
```

Verified installed packages:

```text
transformer_engine 2.13.0
transformer_engine_torch 2.13.0
transformer_engine_cu12 2.13.0
```

## 9. Install Flash Attention

This is the longest build. On the verified H200 pod, it built from source and produced a wheel around 184 MB.

```bash
cd /workspace/lyra/Lyra-2
MAX_JOBS=16 pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3
```

Expected:

```text
Successfully built flash-attn
Successfully installed flash-attn-2.6.3
```

## 10. Build VIPE CUDA Extension

For a repeat H200 install, set `TORCH_CUDA_ARCH_LIST=9.0` to avoid compiling unnecessary CUDA architectures.

During the live install, VIPE was built without this variable and succeeded, but it compiled many architectures and took longer.

```bash
cd /workspace/lyra/Lyra-2
export TORCH_CUDA_ARCH_LIST="9.0"
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
```

Expected package name:

```text
vipe-0.1.1+pt27cu128
```

## 11. Build Depth Anything 3 and gsplat

The first attempt failed because `hatchling` needed `pathspec` for editable metadata. Install the missing build helpers first.

```bash
pip install pathspec pluggy trove-classifiers setuptools-scm
```

Then build Depth Anything 3 and its GS dependencies:

```bash
cd /workspace/lyra/Lyra-2
export TORCH_CUDA_ARCH_LIST="9.0"
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'
```

Notes from the verified install:

- `gsplat` built successfully from source.
- `depth-anything-3` required `numpy<2`, so pip downgraded numpy from `2.2.6` to `1.26.4`.
- Runtime imports still passed after the downgrade.

## 12. Pin gdown for VIPE

VIPE calls `gdown.download(..., fuzzy=...)`. `gdown 6.0.0` removed or changed that argument and reconstruction failed with:

```text
TypeError: download() got an unexpected keyword argument 'fuzzy'
```

Fix:

```bash
pip install "gdown<6"

python - <<'PY'
import gdown, inspect
print("gdown", gdown.__version__)
print(inspect.signature(gdown.download))
PY
```

Expected:

```text
gdown 5.2.2
... fuzzy=False ...
```

## 13. Verify Installation

```bash
cd /workspace/lyra/Lyra-2
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /root/conda-envs/lyra2

export CUDA_HOME=$CONDA_PREFIX
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

PYTHONPATH=. python - <<'PY'
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("all imports OK")
PY

PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference --help >/root/lyra2_zoomgs_help.txt
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon --help >/root/lyra2_recon_help.txt
```

Expected:

```text
torch: 2.7.1+cu128 | cuda: True
gpu: NVIDIA H200 NVL
all imports OK
```

## 14. Download Checkpoints

The model repository was public during the verified run, so Hugging Face login was not required.

If Hugging Face later requires auth, run this on the remote pod and paste the token only into that remote prompt:

```bash
huggingface-cli login
```

Do not paste Hugging Face tokens into Jira, chat, or commit history.

### Verified Live-Session Download Flow

The full checkpoint set is large. On the verified pod, downloading all files into `/workspace` hit a practical storage quota while writing Hugging Face cache metadata, even though `df -h` showed a very large network volume.

To avoid that during the live setup, the main model shards were kept in `/workspace`, but the large reconstruction checkpoint was placed under `/root` and symlinked back.

This is not stop/recreate-safe if RunPod resets `/root`. If `/root` disappears, `checkpoints/recon/model.pt` becomes a broken symlink and reconstruction must be repaired before use.

```bash
cd /workspace/lyra/Lyra-2
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /root/conda-envs/lyra2

# Download everything except the large DA3 reconstruction checkpoint.
huggingface-cli download nvidia/Lyra-2.0 \
  --include "checkpoints/*" \
  --exclude "checkpoints/recon/model.pt" \
  --local-dir .

# Download the large reconstruction checkpoint to local root storage.
mkdir -p /root/lyra2_recon_download
huggingface-cli download nvidia/Lyra-2.0 \
  --include "checkpoints/recon/model.pt" \
  --local-dir /root/lyra2_recon_download

# Symlink it to the path Lyra expects.
mkdir -p /workspace/lyra/Lyra-2/checkpoints/recon
ln -sfn /root/lyra2_recon_download/checkpoints/recon/model.pt \
  /workspace/lyra/Lyra-2/checkpoints/recon/model.pt
```

Verify checkpoint layout:

```bash
cd /workspace/lyra/Lyra-2
du -sh checkpoints /root/lyra2_recon_download
find -L checkpoints -type f | wc -l
find checkpoints -xtype l | wc -l
ls -lah checkpoints/recon/model.pt
```

Expected from the verified run:

```text
checkpoint files: 75
broken symlinks: 0
checkpoints: about 78G
/root/lyra2_recon_download: about 13G
```

### More Persistent Checkpoint Option

If the attached `/workspace` volume is persistent and has enough quota, store `checkpoints/recon/model.pt` directly under `/workspace` instead of symlinking to `/root`.

This avoids the Hugging Face local-dir cache duplication and writes only the final file plus a temporary `.part` file:

```bash
cd /workspace/lyra/Lyra-2
mkdir -p checkpoints/recon
rm -f checkpoints/recon/model.pt checkpoints/recon/model.pt.part

curl -L --fail --retry 10 --retry-delay 2 --connect-timeout 20 \
  -o checkpoints/recon/model.pt.part \
  "https://huggingface.co/nvidia/Lyra-2.0/resolve/main/checkpoints/recon/model.pt?download=true"

mv checkpoints/recon/model.pt.part checkpoints/recon/model.pt
```

Use this option for a teammate handoff if the pod may be stopped later. Use the `/root` symlink option only when `/workspace` quota prevents the direct persistent file.

### Recovery if You Already Hit Disk Quota

This is the exact recovery used during the live setup after `huggingface-cli download ... --include "checkpoints/*"` failed with:

```text
OSError: [Errno 122] Disk quota exceeded
```

Move the partial `recon/model.pt` download from workspace cache to `/root`, remove the workspace cache, resume the single-file download under `/root`, then symlink it back:

```bash
cd /workspace/lyra/Lyra-2

mkdir -p /root/lyra2_recon_download/.cache/huggingface/download/checkpoints

if [ -d .cache/huggingface/download/checkpoints/recon ]; then
  mv .cache/huggingface/download/checkpoints/recon \
    /root/lyra2_recon_download/.cache/huggingface/download/checkpoints/
fi

rm -rf .cache/huggingface
mkdir -p checkpoints/recon

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /root/conda-envs/lyra2

huggingface-cli download nvidia/Lyra-2.0 \
  --include "checkpoints/recon/model.pt" \
  --local-dir /root/lyra2_recon_download

ln -sfn /root/lyra2_recon_download/checkpoints/recon/model.pt \
  /workspace/lyra/Lyra-2/checkpoints/recon/model.pt
```

## 15. Smoke Test: Generate a DMD Video

This is the exact smoke test that passed on the verified RunPod pod.

```bash
cd /workspace/lyra/Lyra-2
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /root/conda-envs/lyra2

export CUDA_HOME=$CONDA_PREFIX
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference \
  --input_image_path assets/samples \
  --sample_id 4 \
  --experiment lyra2 \
  --checkpoint_dir checkpoints/model \
  --prompt_dir assets/samples \
  --output_path outputs/zoomgs_smoke \
  --num_frames_zoom_in 81 \
  --num_frames_zoom_out 81 \
  --use_dmd
```

Expected outputs:

```text
outputs/zoomgs_smoke/04/zoom_in.mp4
outputs/zoomgs_smoke/04/zoom_out.mp4
outputs/zoomgs_smoke/videos/04.mp4
```

Verified output sizes:

```text
outputs/zoomgs_smoke/videos/04.mp4                about 3.1M
outputs/zoomgs_smoke/04/zoom_in.mp4               about 1.9M
outputs/zoomgs_smoke/04/zoom_out.mp4              about 2.4M
```

Runtime notes:

- First run spent most of its time loading the 64 GB model checkpoint shards from the `/workspace` network volume.
- After checkpoint load, DMD generation for `81 + 81` frames completed successfully.
- Peak model memory observed during generation was roughly 68 GB before sampling, well under H200 capacity.

## 16. Smoke Test: Gaussian Splat Reconstruction

Run reconstruction on the generated video:

```bash
cd /workspace/lyra/Lyra-2
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /root/conda-envs/lyra2

export CUDA_HOME=$CONDA_PREFIX
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon \
  --input_video_path outputs/zoomgs_smoke/videos/04.mp4
```

The first reconstruction run downloads additional VIPE assets into `/root/.cache/torch/hub`:

```text
geocalib-pinhole.tar  about 111M
droid.pth            about 16M
```

Expected outputs:

```text
outputs/zoomgs_smoke/videos/04_gs_ours/reconstructed_scene.ply
outputs/zoomgs_smoke/videos/04_gs_ours/gs_trajectory.mp4
outputs/zoomgs_smoke/videos/04_gs_ours/cameras.npz
outputs/zoomgs_smoke/videos/04_gs_ours/vipe_predictions.npz
outputs/zoomgs_smoke/videos/04_gs_ours/.done
```

Verified output sizes:

```text
outputs/zoomgs_smoke/videos/04_gs_ours/reconstructed_scene.ply  about 758M
outputs/zoomgs_smoke/videos/04_gs_ours/gs_trajectory.mp4        about 1.8M
outputs/zoomgs_smoke/videos/04_gs_ours                          about 760M total
```

## 17. Copy Small Outputs Back to Mac

Example:

```bash
mkdir -p /Users/<LOCAL_USER>/Documents/3D/lyra/runpod_outputs/lyra2_smoke

scp -i ~/.ssh/id_ed25519 -P <RUNPOD_PORT> \
  root@<RUNPOD_IP>:/workspace/lyra/Lyra-2/outputs/zoomgs_smoke/videos/04.mp4 \
  /Users/<LOCAL_USER>/Documents/3D/lyra/runpod_outputs/lyra2_smoke/04_combined.mp4

scp -i ~/.ssh/id_ed25519 -P <RUNPOD_PORT> \
  root@<RUNPOD_IP>:/workspace/lyra/Lyra-2/outputs/zoomgs_smoke/videos/04_gs_ours/gs_trajectory.mp4 \
  /Users/<LOCAL_USER>/Documents/3D/lyra/runpod_outputs/lyra2_smoke/04_gs_trajectory.mp4
```

The verified local files were:

```text
runpod_outputs/lyra2_smoke/04_combined.mp4       about 3.1M
runpod_outputs/lyra2_smoke/04_gs_trajectory.mp4  about 1.8M
```

The `.ply` file is much larger. Copy it only if needed:

```bash
scp -i ~/.ssh/id_ed25519 -P <RUNPOD_PORT> \
  root@<RUNPOD_IP>:/workspace/lyra/Lyra-2/outputs/zoomgs_smoke/videos/04_gs_ours/reconstructed_scene.ply \
  /Users/<LOCAL_USER>/Documents/3D/lyra/runpod_outputs/lyra2_smoke/
```

## 18. Resume Work After SSH Reconnect or Pod Restart

```bash
ssh root@<RUNPOD_IP> -p <RUNPOD_PORT> -i ~/.ssh/id_ed25519
```

First check whether this is the same installed pod or a reset/recreated pod:

```bash
test -x /root/conda-envs/lyra2/bin/python && echo "lyra2 env exists" || echo "lyra2 env missing"
test -d /workspace/lyra/Lyra-2 && echo "Lyra-2 repo exists" || echo "Lyra-2 repo missing"
test -f /workspace/lyra/Lyra-2/checkpoints/model/model/.metadata && echo "model checkpoint metadata exists" || echo "model checkpoints missing"
find /workspace/lyra/Lyra-2/checkpoints -xtype l 2>/dev/null || true
```

Interpretation:

- If `lyra2 env exists`, this is a same-session or persistent-root resume. Continue with the activation commands below.
- If `lyra2 env missing`, `/root` was reset. Rerun sections 4 through 12 before trying inference.
- If `Lyra-2 repo missing`, `/workspace` was also reset or a different volume is attached. Rerun from section 2 or reattach the correct RunPod volume.
- If `find ... -xtype l` prints `checkpoints/recon/model.pt`, the symlink points to a missing `/root/lyra2_recon_download` file. Redownload the reconstruction checkpoint or store it directly under `/workspace`.

For a healthy same-pod resume:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /root/conda-envs/lyra2
cd /workspace/lyra/Lyra-2

export CUDA_HOME=$CONDA_PREFIX
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Quick health check:

```bash
PYTHONPATH=. python - <<'PY'
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("imports OK")
PY

find -L checkpoints -type f | wc -l
find checkpoints -xtype l | wc -l
```

Expected:

```text
imports OK
checkpoint files: 75
broken symlinks: 0
```

If `conda activate /root/conda-envs/lyra2` fails after RunPod Stop/Recreate, do not debug the inference command yet. The environment is gone; rebuild it first.

## 19. Safe Cleanup After Install

These caches can be removed after the environment and checkpoints are installed:

```bash
rm -rf /root/conda-pkgs/* /root/.cache/pip/*
```

Do not remove these during an active pod session unless you intentionally want to reinstall:

```text
/root/conda-envs/lyra2
/workspace/miniforge3
/workspace/lyra
/workspace/lyra/Lyra-2/checkpoints
/root/lyra2_recon_download
```

`/root/lyra2_recon_download` must remain if `checkpoints/recon/model.pt` is a symlink to it.

Persistence note: files under `/root` may disappear after a RunPod Stop/Recreate depending on the pod configuration. Files under `/workspace` are only safe if the same persistent RunPod volume remains attached.

Verified disk state after cleanup:

```text
/root/conda-envs/lyra2         about 18G
/root/lyra2_recon_download     about 13G
/workspace/lyra/Lyra-2         about 80G
/root/conda-pkgs               0
/root/.cache/pip               0
```

## 20. Troubleshooting Summary

Issues encountered and fixes applied:

1. `mamba package extract` stuck in `D` state under `/workspace/conda-pkgs`
   - Fix: use `/root/conda-envs` and `/root/conda-pkgs`.

2. Accidental CUDA newer than 12.8 when CUDA was not pinned
   - Fix: recreate env and install `cuda=12.8` from `nvidia/label/cuda-12.8.0`.

3. `conda activate` failed under `set -u` with `SYS_SYSROOT: unbound variable`
   - Fix: do not use `set -u` around conda activation.

4. Transformer Engine failed with missing `nvtx3/nvToolsExt.h`
   - Fix: add `site-packages/nvidia/nvtx/include` to include paths and symlink NVTX headers into `$CONDA_PREFIX/include`.

5. `depth_anything_3` editable metadata failed with missing `pathspec`
   - Fix: install `pathspec pluggy trove-classifiers setuptools-scm`.

6. `gdown 6.0.0` failed with `download() got an unexpected keyword argument 'fuzzy'`
   - Fix: `pip install "gdown<6"`.

7. Hugging Face checkpoint download hit workspace quota
   - Live-session fix: store `checkpoints/recon/model.pt` under `/root/lyra2_recon_download` and symlink it into `checkpoints/recon/model.pt`.
   - Persistence fix: store `checkpoints/recon/model.pt` directly under `/workspace/lyra/Lyra-2/checkpoints/recon/model.pt` using the direct `curl` flow in section 14.

8. Resolver warnings about optional packages
   - Observed warnings included missing `yacs`, `asciitree`, `iniconfig`, `execnet`, and some test/dev dependencies.
   - These did not block verified inference or reconstruction.

9. RunPod Stop/Recreate removed `/root`
   - Effect: `/root/conda-envs/lyra2` disappeared, so `conda activate /root/conda-envs/lyra2` failed.
   - Effect: any symlink from `checkpoints/recon/model.pt` to `/root/lyra2_recon_download/...` became broken.
   - Fix: rerun sections 4 through 12 to rebuild the env, then repair checkpoints using section 14.

10. PyTorch reports `cuda available: False`
   - Verified failure state: `/dev/nvidia*` was empty, `nvidia-smi` was not installed, and `env | grep NVIDIA` returned nothing.
   - This is a RunPod runtime/pod allocation problem, not a Lyra problem.
   - Fix: start or recreate a GPU-backed RunPod pod, connect to the new SSH port, and verify `nvidia-smi` before running any Lyra command.

11. RunPod says `There are no instances currently available`
   - This means the selected GPU class is not available for automatic migration at that moment.
   - Do not choose CPU mode for Lyra inference. CPU mode is only useful for data access or backup.
   - Safe options:
     - Wait and retry automatic migration later.
     - Deploy a new GPU pod with a different compatible GPU type, preferably H200, H100, or A100 80GB.
     - If using a network volume, attach the same volume to the new GPU pod.
     - If not using a network volume, do not terminate the current pod until `/workspace/lyra/Lyra-2/checkpoints` and needed outputs are backed up or migrated.
   - After the new GPU pod starts, verify `nvidia-smi`, `/dev/nvidia*`, and the checkpoint shard count before rebuilding the conda environment.
