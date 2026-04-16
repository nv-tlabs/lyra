# Lyra 2.0: Explorable Generative 3D Worlds

<video src="https://github.com/user-attachments/assets/f47c24c0-453e-4134-84f1-80c56613f4af" autoplay controls loop muted playsinline width="720"></video>

**Lyra 2.0: Explorable Generative 3D Worlds**<br>
[Tianchang Shen\*](https://www.cs.toronto.edu/~shenti11/),
[Sherwin Bahmani](https://sherwinbahmani.github.io/),
[Kai He](https://www.cs.toronto.edu/~hekai/),
[Sangeetha Grama Srinivasan](https://pages.cs.wisc.edu/~sgsrinivasa2/),
[Tianshi Cao](https://www.linkedin.com/in/tianshi-cao-a23270b1/?originalSubdomain=ca),
[Jiawei Ren](https://jiawei-ren.github.io/),
[Ruilong Li](https://www.liruilong.cn/),
[Zian Wang](https://www.cs.toronto.edu/~zianwang/),
[Nicholas Sharp](https://nmwsharp.com/),
[Zan Gojcic](https://zgojcic.github.io/),
[Sanja Fidler](https://www.cs.toronto.edu/~fidler/),
[Jiahui Huang](https://huangjh-pub.github.io/),
[Huan Ling](https://www.cs.toronto.edu/~linghuan/),
[Jun Gao](https://www.cs.toronto.edu/~jungao/),
[Xuanchi Ren\*](https://xuanchiren.com/) <br>
\* indicates equal contribution<br>
NVIDIA <br>
**[Paper](https://arxiv.org/abs/2604.13036), [Project Page](https://research.nvidia.com/labs/sil/projects/lyra2/), [HuggingFace](https://huggingface.co/nvidia/Lyra-2.0)**

**TL;DR: Lyra 2.0 turns an image into a 3D world you can walk through, look back, and drop a robot into for real-time rendering, simulation, and immersive applications.**

**Abstract**: Recent advances in video generation enable a new paradigm for 3D scene creation: generating camera-controlled videos that simulate scene walkthroughs, then lifting them to 3D via feed-forward reconstruction techniques. This generative reconstruction approach combines the visual fidelity and creative capacity of video models with 3D outputs ready for real-time rendering and simulation. Scaling to large, complex environments requires 3D-consistent video generation over long camera trajectories with large viewpoint changes and location revisits, a setting where current video models degrade quickly. Existing methods for long-horizon generation are fundamentally limited by two forms of degradation: spatial forgetting and temporal drifting. As exploration proceeds, previously observed regions fall outside the model's temporal context, forcing the model to hallucinate structures when revisited. Meanwhile, autoregressive generation accumulates small synthesis errors over time, gradually distorting scene appearance and geometry. We present Lyra 2.0, a framework for generating persistent, explorable 3D worlds at scale. To address spatial forgetting, we maintain per-frame 3D geometry and use it solely for information routing—retrieving relevant past frames and establishing dense correspondences with the target viewpoints—while relying on the generative prior for appearance synthesis. To address temporal drifting, we train with self-augmented histories that expose the model to its own degraded outputs, teaching it to correct drift rather than propagate it. Together, these enable substantially longer and 3D-consistent video trajectories, which we leverage to fine-tune feed-forward reconstruction models that reliably recover high-quality 3D scenes.


## News

- 🚀 [April 15, 2026] Paper, model weights, and inference code are now publicly available!
- 🔜 [Coming Soon] GUI for interactive scene generation.
- 🔜 [Coming Soon] 4-step distillation LoRA.

## Installation

See **[INSTALL.md](INSTALL.md)** for full step-by-step instructions.


## Inference

### Download Checkpoints

Download the pretrained checkpoints from [HuggingFace](https://huggingface.co/nvidia/Lyra-2.0):

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download all checkpoints into the checkpoints/ directory
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
```

### Step 1 — Generate Video

Lyra 2.0 takes a starting image, a user-defined camera trajectory, and per-chunk text captions as input to generate an exploration video. For ease of testing, we provide several options for video generation.

#### Option 1: Preset Trajectory (Zoom-in / Zoom-out)

Generate a zoom-in then zoom-out exploration video from a single input image. Because outpainting is gradual in this mode, a single text prompt describing the starting image is sufficient—each image needs a paired `.txt` file with its caption in the same directory (see `assets/samples/` for examples). Change `--sample_id` (0–14) to run different examples.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference \
  --input_image_path assets/samples \
  --sample_id 4 \
  --experiment lyra2 \
  --checkpoint_dir checkpoints/model \
  --prompt_dir assets/samples \
  --output_path outputs/zoomgs \
  --num_frames_zoom_in 81 \
  --num_frames_zoom_out 241 \
  --zoom_in_strength 0.5 \
  --zoom_out_strength 1.5
```

Outputs written to `outputs/zoomgs/<sample_id>/`:
- `zoom_in.mp4` — zoom-in clip
- `zoom_out.mp4` — zoom-out clip
- `videos/<sample_id>.mp4` — combined video (zoom-in → zoom-out)

**Tips:**
- The scale of the zoom-in and zoom-out trajectories can be adjusted via `--zoom_in_strength` and `--zoom_out_strength`. Note that a built-in collision detection mechanism prevents the camera from moving into objects, so increasing `--zoom_in_strength` may not always change the trajectory.
- If the camera movement appears too fast, increase `--num_frames_zoom_in` or `--num_frames_zoom_out` to spread the motion over more frames. Frame counts must be of the form 1 + 80k (i.e., 81, 161, 241, …) to align with autoregressive chunk boundaries.

**Runtime on 1× H100 80GB:** ~9 min per 80 frames.

**Example outputs (Step 1 video generation | Step 2 GS reconstruction):**

<table><tr>
<td><video src="https://github.com/user-attachments/assets/12cb922d-da2a-4ede-9bfb-eb970213951a" autoplay controls loop muted playsinline preload="auto" width="360"></video></td>
<td><video src="https://github.com/user-attachments/assets/0d44e0fa-e7e3-4bd2-af42-3bd2ec7dffc0" autoplay controls loop muted playsinline preload="auto" width="360"></video></td>
</tr></table>

#### Option 2: Custom Trajectory

For evaluating Lyra 2.0 on a custom, predefined camera trajectory, prepare your inputs following the examples under `assets/custom_trajectory_examples/`. Each example folder contains:
- `first_frame.png` — the starting image
- `trajectory.npz` — camera poses (`w2c`: N×4×4 world-to-camera matrices, `intrinsics`: N×3×3, `image_height`, `image_width`)
- `captions.json` — per-chunk text captions keyed by frame index (e.g. `{"0": "...", "81": "...", ...}`)

The script loads the first `--num_frames` poses from the trajectory file and runs FramePack AR generation. When `--captions_path` is provided with a multi-entry JSON, each AR chunk uses the caption whose frame index is closest to (but not exceeding) the current chunk start.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. python -m lyra_2._src.inference.lyra2_custom_traj_inference \
  --input_image_path assets/custom_trajectory_examples/example_0/first_frame.png \
  --trajectory_path assets/custom_trajectory_examples/example_0/trajectory.npz \
  --experiment lyra2 \
  --checkpoint_dir checkpoints/model \
  --captions_path assets/custom_trajectory_examples/example_0/captions.json \
  --num_frames 481 \
  --output_path outputs/custom_traj
```

Output: `outputs/custom_traj/<image_name>.mp4`

> **Note:** The model may hallucinate objects along the predefined trajectory, which can lead to degenerated results. Additionally, the input trajectory is always relative to the monocular depth estimated from the first frame, so the actual camera path may differ from what you expect. You can adjust `--pose_scale` to scale your trajectory.

**Example outputs (Step 1 video generation | Step 2 GS reconstruction):**

<table><tr>
<td><video src="https://github.com/user-attachments/assets/61354390-cb6f-484b-87c6-f7ef7ca466f2" autoplay controls loop muted playsinline preload="auto" width="360"></video></td>
<td><video src="https://github.com/user-attachments/assets/61cfe2f0-b0bf-4077-8367-49968aca7a19" autoplay controls loop muted playsinline preload="auto" width="360"></video></td>
</tr></table>

#### Option 3: Interactive GUI (Coming Soon!)

We will release our interactive GUI, an online captioning pipeline, and an instruction video in an upcoming update. Stay tuned!

<video src="https://github.com/user-attachments/assets/76394f4d-b6c7-46f2-8133-5eabf5fd74e1" autoplay controls loop muted playsinline preload="auto" width="360"></video>

### Step 2 — 3D Gaussian Splatting Reconstruction

Lift the generated video to a 3D Gaussian Splatting scene using VIPE pose estimation and DA3 depth.

```bash
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon \
  --input_video_path outputs/zoomgs/videos/<sample_id>.mp4
```

Output is written next to the input video as `<sample_id>_gs_ours/`:
- `reconstructed_scene.ply` — Gaussian point cloud
- `gs_trajectory.mp4` — rendered flythrough video

**Runtime on 1× H100 80GB:** ~1 min (VIPE + DA3 depth + GS render).


## Acknowledgement

We thank the following open-source projects that this work builds upon:
- [Wan 2.1](https://github.com/Wan-Video/Wan2.1)
- [ChronoEdit](https://github.com/nv-tlabs/ChronoEdit)
- [GenWarp](https://github.com/sony/genwarp)
- [FusionX LoRA](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX)
- [FramePack](https://github.com/lllyasviel/FramePack)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [MoGe](https://github.com/microsoft/MoGe)

## Citation

```bibtex
@article{shen2026lyra2,
    title={Lyra 2.0: Explorable Generative 3D Worlds},
    author={Shen, Tianchang and Bahmani, Sherwin and He, Kai and Srinivasan, Sangeetha Grama and Cao, Tianshi and Ren, Jiawei and Li, Ruilong and Wang, Zian and Sharp, Nicholas and Gojcic, Zan and Fidler, Sanja and Huang, Jiahui and Ling, Huan and Gao, Jun and Ren, Xuanchi},
    journal={arXiv preprint arXiv:2604.13036},
    year={2026}
}
```

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

Lyra 2.0 source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

Lyra 2.0 models are released under the [NVIDIA Internal Scientific Research and Development Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/). For a custom license, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
