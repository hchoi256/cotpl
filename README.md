# CoT-PL: Visual Chain-of-Thought Reasoning Meets Pseudo-Labeling for Open-Vocabulary Object Detection

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aligning-bag-of-regions-for-open-vocabulary/open-vocabulary-object-detection-on-mscoco&#41;]&#40;https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=aligning-bag-of-regions-for-open-vocabulary&#41;)

## Introduction

This is an official release of the paper **CoT-PL: Visual Chain-of-Thought Reasoning Meets Pseudo-Labeling for Open-Vocabulary Object Detection**.

> [**CoT-PL: Visual Chain-of-Thought Reasoning Meets Pseudo-Labeling for Open-Vocabulary Object Detection**](https://arxiv.org/abs/2510.14792),            
> Hojun Choi, Youngsun Lim, Jaeyo Shin, Hyunjung Shim
<!-- > In: Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023            -->
> [[Paper](https://arxiv.org/pdf/2510.14792)][[project page(TBD)]()][[Bibetex](https://github.com/wusize/ovdet#citation)]


## Updates

⛽⛽⛽ Contact: eric970412@gmail.com

- [✅] [2024.12.31] 👨‍💻 The official codes have been released!
<!-- - [✅] [2024.12.31] 🎉 Our paper has been accepted to [AAAI 2025](https://openreview.net/group?id=AAAI.org/2025/Conference#tab-accept)! -->
- [✅] [2024.10.16] 📄 Our paper is now available! You can find the paper [here](https://arxiv.org/abs/2510.14792).



## Installation

This project is based on [MMDetection 3.x](https://github.com/open-mmlab/mmdetection/tree/3.x)

It requires the following OpenMMLab packages:

- MMEngine >= 0.6.0
- MMCV-full >= v2.0.0rc4
- MMDetection >= v3.0.0rc6
- lvisapi

```bash
pip install openmim mmengine
mim install "mmcv>=2.0.0rc4"
pip install git+https://github.com/lvis-dataset/lvis-api.git
mim install "mmdet>=3.0.0rc6"
pip install ftfy regex
```

## License

This project is released under the [NTU S-Lab License 1.0](LICENSE).



## Usage
### Obtain CLIP Checkpoints
We use CLIP's ViT-B-16 model for the implementation of our method.
`pip install git+https://github.com/openai/CLIP.git` and run 
```python
import clip
import torch
model, _ = clip.load("ViT-B/16")
torch.save(model.state_dict(), 'checkpoints/clip_vitb16.pth')
```

### Pseudo-Label Generation

The pseudo-label generation is on [pseudo-label](pseudo-labels/README.md) or download `instances_train2017_pseudo_v0_new.json` from [huggingface](https://huggingface.co/datasets/hchoi256/CoT-PL/tree/main).


### Training and Testing

The training and testing on [OV-COCO](configs/baron/ov_coco/README.md) are supported now.


## Citation

```bibtex
@misc{choi2025cotplvisualchainofthoughtreasoning,
      title={CoT-PL: Visual Chain-of-Thought Reasoning Meets Pseudo-Labeling for Open-Vocabulary Object Detection}, 
      author={Hojun Choi and Youngsun Lim and Jaeyo Shin and Hyunjung Shim},
      year={2025},
      eprint={2510.14792},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14792}, 
}
```

