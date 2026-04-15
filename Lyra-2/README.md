# Lyra 2.0: Explorable Generative 3D Worlds

🚧 **Under Construction** — Code release today. Stay tuned!

📄 [Paper](https://arxiv.org/abs/2604.13036) | 🌐 [Project Page](https://research.nvidia.com/labs/sil/projects/lyra2/) | 🤗 [HuggingFace]()

**TL;DR: Lyra 2.0 turns an image into a 3D world you can walk through, look back, and drop a robot into for real-time rendering, simulation, and immersive applications.**

**Abstract**: Recent advances in video generation enable a new paradigm for 3D scene creation: generating camera-controlled videos that simulate scene walkthroughs, then lifting them to 3D via feed-forward reconstruction techniques. This generative reconstruction approach combines the visual fidelity and creative capacity of video models with 3D outputs ready for real-time rendering and simulation. Scaling to large, complex environments requires 3D-consistent video generation over long camera trajectories with large viewpoint changes and location revisits, a setting where current video models degrade quickly. Existing methods for long-horizon generation are fundamentally limited by two forms of degradation: spatial forgetting and temporal drifting. As exploration proceeds, previously observed regions fall outside the model's temporal context, forcing the model to hallucinate structures when revisited. Meanwhile, autoregressive generation accumulates small synthesis errors over time, gradually distorting scene appearance and geometry. We present Lyra 2.0, a framework for generating persistent, explorable 3D worlds at scale. To address spatial forgetting, we maintain per-frame 3D geometry and use it solely for information routing—retrieving relevant past frames and establishing dense correspondences with the target viewpoints—while relying on the generative prior for appearance synthesis. To address temporal drifting, we train with self-augmented histories that expose the model to its own degraded outputs, teaching it to correct drift rather than propagate it. Together, these enable substantially longer and 3D-consistent video trajectories, which we leverage to fine-tune feed-forward reconstruction models that reliably recover high-quality 3D scenes.


[Tianchang Shen](https://www.cs.toronto.edu/~shenti11/)\*,
[Sherwin Bahmani](https://sherwinbahmani.github.io/)\*,
[Kai He](https://www.cs.toronto.edu/~hekai/),
[Sangeetha Grama Srinivasan](https://pages.cs.wisc.edu/~sgsrinivasa2/),
[Tianshi Cao](https://scholar.google.com/citations?user=CZ9wBBoAAAAJ&hl=en),
[Jiawei Ren](https://jiawei-ren.github.io/),
[Ruilong Li](https://www.liruilong.cn/),
[Zian Wang](https://www.cs.toronto.edu/~zianwang/),
[Nicholas Sharp](https://nmwsharp.com/),
[Zan Gojcic](https://zgojcic.github.io/),
[Sanja Fidler](https://www.cs.utoronto.ca/~fidler/),
[Jiahui Huang](https://huangjh-pub.github.io/),
[Huan Ling](https://www.cs.toronto.edu/~linghuan/),
[Jun Gao](https://www.cs.toronto.edu/~jungao/),
[Xuanchi Ren](https://xuanchiren.com/)\* <br>

\* Equal Contribution

## Citation

```bibtex
@article{shen2026lyra2,
  title={Lyra 2.0: Explorable Generative 3D Worlds},
  author={Shen, Tianchang and Bahmani, Sherwin and He, Kai and Srinivasan, Sangeetha Grama and Cao, Tianshi and 
          Ren, Jiawei and Li, Ruilong and Wang, Zian and Sharp, Nicholas and Gojcic, Zan and Fidler, Sanja and 
          Huang, Jiahui and Ling, Huan and Gao, Jun and Ren, Xuanchi},
  journal={arXiv preprint arXiv:2604.13036},
  year={2026}
}
```

## License

Lyra 2.0 source code is released under the [Apache 2.0 License](../LICENSE). 
