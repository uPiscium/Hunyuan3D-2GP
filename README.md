[中文阅读](README_zh_cn.md)
[日本語で読む](README_ja_jp.md)

# Hunyuan3D-2GP: 3D Generation for the GPU Poor
*GPU Poor version by **DeepBeepMeep**. This great video generator can now run smoothly with less than 6 GB of VRAM.*
<BR>

This is another integration of the *mmgp 3.1* module that allows easy to setup advanced and fast offloading.\
https://github.com/deepbeepmeep/mmgp

<div align="center">
  <a href=https://3d.hunyuan.tencent.com target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/spaces/tencent/Hunyuan3D-2  target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href=https://huggingface.co/tencent/Hunyuan3D-2 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://3d-models.hunyuan.tencent.com/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://discord.gg/GuaWYwzKbX target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href=https://arxiv.org/abs/2501.12202 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://x.com/txhunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
 <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px></a>
</div>

[//]: # (  <a href=# target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>)

[//]: # (  <a href=# target="_blank"><img src= https://img.shields.io/badge/Colab-8f2628.svg?logo=googlecolab height=22px></a>)

[//]: # (  <a href="#"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/v/mulankit?logo=pypi"  height=22px></a>)

<br>

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/GuaWYwzKbX)** group to discuss and find help from us.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> |        

---

<p align="center">
“ Living out everyone’s imagination on creating and manipulating 3D assets.”
</p>

## 🔥 News
- Jan 23, 2025: 💬 Hunyuan3D-2.0GP by Deepbeepmeep: added lighning fix in rendering window
- Jan 23, 2025: 💬 Hunyuan3D-2.0GP by Deepbeepmeep: added Windows support thanks to MrForExample and sdbds + omitted optimization that keeps under VRAM 6GB with profile 4 or 5
- Jan 22, 2025: 💬 Hunyuan3D-2.0GP by Deepbeepmeep: low VRAM support and unlocked text to 3D generator
- Jan 21, 2025: 💬 Release [Hunyuan3D 2.0](https://huggingface.co/spaces/tencent/Hunyuan3D-2). Please give it a try!

- Jan 23, 2025: 💬 We thank community members for creating [Windows installation tool](https://github.com/YanWenKun/Comfy3D-WinPortable/releases/tag/r8-hunyuan3d2), ComfyUI support with [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper) and [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack) and other awesome [extensions](#community-resources).
- Jan 21, 2025: 💬 Enjoy exciting 3D generation on our website [Hunyuan3D Studio](https://3d.hunyuan.tencent.com)!
- Jan 21, 2025: 💬 Release inference code and pretrained models
  of [Hunyuan3D 2.0](https://huggingface.co/tencent/Hunyuan3D-2).
- Jan 21, 2025: 💬 Release Hunyuan3D 2.0. Please give it a try
  via [huggingface space](https://huggingface.co/spaces/tencent/Hunyuan3D-2) and
  our [official site](https://3d.hunyuan.tencent.com)!

## **Abstract**

We present Hunyuan3D 2.0, an advanced large-scale 3D synthesis system for generating high-resolution textured 3D assets.
This system includes two foundation components: a large-scale shape generation model - Hunyuan3D-DiT, and a large-scale
texture synthesis model - Hunyuan3D-Paint.
The shape generative model, built on a scalable flow-based diffusion transformer, aims to create geometry that properly
aligns with a given condition image, laying a solid foundation for downstream applications.
The texture synthesis model, benefiting from strong geometric and diffusion priors, produces high-resolution and vibrant
texture maps for either generated or hand-crafted meshes.
Furthermore, we build Hunyuan3D-Studio - a versatile, user-friendly production platform that simplifies the re-creation
process of 3D assets. It allows both professional and amateur users to manipulate or even animate their meshes
efficiently.
We systematically evaluate our models, showing that Hunyuan3D 2.0 outperforms previous state-of-the-art models,
including the open-source models and closed-source models in geometry details, condition alignment, texture quality, and
e.t.c.

## How to run the Gradio app
1) Follow the installation instructions below

2) Enter either one of the commande lines in bash session

To run the image to 3E generator:
```bash
python gradio_app.py
```

To run the text to 3D generator:
```bash
python gradio_app.py --enable_t23d

```

By default the memory profile assumes 9 GB of VRAM *(profile 2)*. If you have less but at least 6 GB of VRAM add *--profile 5*

To run the image to 3D generator with optimized memory management:
```bash
python gradio_app.py --profile 5

```
To run the text to 3D generator with optimized memory management:
```bash
python gradio_app.py --enable_t23d --profile 5

```

You can choose between 5 profiles depending on your hardware:
- HighRAM_HighVRAM  (1): at least 48 GB of RAM and 12 GB of VRAM 
- HighRAM_LowVRAM  (2): at least 48 GB of RAM and 6 GB of VRAM
- LowRAM_HighVRAM  (3): at least 32 GB of RAM and 12 GB of VRAM
- LowRAM_LowVRAM  (4): at least 32 GB of RAM and 6 GB of VRAM
- VerylowRAM_LowVRAM  (5): at least 24 GB of RAM and 6 GB of VRAM 

Usualy the lower the profile the faster the generation.

## Other GPU Poor Applications
- HuanyuanVideoGP: https://github.com/deepbeepmeep/HunyuanVideoGP\
One of the best open source Text to Video generator

- FluxFillGP: https://github.com/deepbeepmeep/FluxFillGP\
One of the best inpainting / outpainting tools based on Flux that can run with less than 12 GB of VRAM.

- Cosmos1GP: https://github.com/deepbeepmeep/Cosmos1GP\
This application include two models: a text to world generator and a image / video to world (probably the best open source image to video generator).





### Architecture

Hunyuan3D 2.0 features a two-stage generation pipeline, starting with the creation of a bare mesh, followed by the
synthesis of a texture map for that mesh. This strategy is effective for decoupling the difficulties of shape and
texture generation and also provides flexibility for texturing either generated or handcrafted meshes.

<p align="left">
  <img src="assets/images/arch.jpg">
</p>

### Performance

We have evaluated Hunyuan3D 2.0 with other open-source as well as close-source 3d-generation methods.
The numerical results indicate that Hunyuan3D 2.0 surpasses all baselines in the quality of generated textured 3D assets
and the condition following ability.

| Model                   | CMMD(⬇)   | FID_CLIP(⬇) | FID(⬇)      | CLIP-score(⬆) |
|-------------------------|-----------|-------------|-------------|---------------|
| Top Open-source Model1  | 3.591     | 54.639      | 289.287     | 0.787         |
| Top Close-source Model1 | 3.600     | 55.866      | 305.922     | 0.779         |
| Top Close-source Model2 | 3.368     | 49.744      | 294.628     | 0.806         |
| Top Close-source Model3 | 3.218     | 51.574      | 295.691     | 0.799         |
| Hunyuan3D 2.0           | **3.193** | **49.165**  | **282.429** | **0.809**     |

Generation results of Hunyuan3D 2.0:
<p align="left">
  <img src="assets/images/e2e-1.gif"  height=300>
  <img src="assets/images/e2e-2.gif"  height=300>
</p>

### Pretrained Models

| Model                  | Date       | Params | Huggingface                                                                             |
|------------------------|------------|--------|-----------------------------------------------------------------------------------------| 
| Hunyuan3D-DiT-v2-0     | 2025-01-21 | 2.6B   | [Download](https://huggingface.co/tencent/Hunyuan3D-2)                                  |
| Hunyuan3D-Paint-v2-0   | 2025-01-21 | 1.3B   | [Download](https://huggingface.co/tencent/Hunyuan3D-2)                                  |
| Hunyuan3D-Delight-v2-0 | 2025-01-21 | 1.3B   | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-delight-v2-0) | 

## 🤗 Get Started with Hunyuan3D 2.0

You may follow the next steps to use Hunyuan3D 2.0 via code or the Gradio App.

### Install Requirements
To use the application on Windows (without WSL) you will need to install Microsoft Visual Studio 2022 or later. If you get an error during the execution of onr of the *python setup.py* below you will need to set the path to the C++ compiler by running the following script (once you have located the installation path of VS Studio which may differ):
```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd" -arch=x64 
```


In any case please make sure you have Python 3.10 installed, you may create a conda environnment: 
```bash
conda create -n Hunyuan3D-2GP python==3.10.9 
```
Then install the required libraries:
```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
pip install -r requirements.txt
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
bash compile_mesh_painter.sh OR python3 setup.py install (on Windows)
```

### API Usage

We designed a diffusers-like API to use our shape generation model - Hunyuan3D-DiT and texture synthesis model -
Hunyuan3D-Paint.

You could assess **Hunyuan3D-DiT** via:

```python
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]
```

The output mesh is a [trimesh object](https://trimesh.org/trimesh.html), which you could save to glb/obj (or other
format) file.

For **Hunyuan3D-Paint**, do the following:

```python
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# let's generate a mesh first
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/demo.png')
```

Please visit [minimal_demo.py](minimal_demo.py) for more advanced usage, such as **text to 3D** and **texture generation
for handcrafted mesh**.

### Gradio App

You could also host a [Gradio](https://www.gradio.app/) App in your own computer via:

```bash
pip3 install gradio==3.39.0
python3 gradio_app.py
```

Don't forget to visit [Hunyuan3D](https://3d.hunyuan.tencent.com) for quick use, if you don't want to host yourself.

## 📑 Open-Source Plan

- [x] Inference Code
- [x] Model Checkpoints
- [ ] ComfyUI
- [ ] TensorRT Version

## 🔗 BibTeX

If you found this repository helpful, please cite our report:

```bibtex
@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{yang2024hunyuan3d,
    title={Hunyuan3D 1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation},
    author={Tencent Hunyuan3D Team},
    year={2024},
    eprint={2411.02293},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Community Resources

Thanks for the contributions of community members, here we have these great extensions of Hunyuan3D 2.0:

- [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack) 
- [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)
- [Hunyuan3D-2-for-windows](https://github.com/sdbds/Hunyuan3D-2-for-windows)
- [📦 A bundle for running on Windows | 整合包](https://github.com/YanWenKun/Comfy3D-WinPortable/releases/tag/r8-hunyuan3d2)


## Acknowledgements

We would like to thank the contributors to
the [DINOv2](https://github.com/facebookresearch/dinov2), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D),
and [Michelangelo](https://github.com/NeuralCarver/Michelangelo/tree/main) repositories, for their open research and
exploration.

## Star History

<a href="https://star-history.com/#Tencent/Hunyuan3D-2&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2&type=Date" />
 </picture>
</a>
