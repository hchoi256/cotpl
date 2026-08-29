# [ECCV 26] MSPL: Multi-Step Pseudo-Labeling for Open-Vocabulary Object Detection

<img src="./assets/eccv.png" alt="스크린샷" width="500">

## Introduction

This is an official release of the paper **MSPL: Multi-Step Pseudo-Labeling for Open-Vocabulary Object Detection**.

> [**MSPL: Multi-Step Pseudo-Labeling for Open-Vocabulary Object Detection**](https://arxiv.org/abs/2510.14792),            
> Hojun Choi, Youngsun Lim, Jaeyo Shin, Hyunjung Shim
> [[Paper](https://arxiv.org/pdf/2510.14792)][[Project Page](https://hchoi256.github.io/projects/mspl/)][[Dataset](https://huggingface.co/datasets/hchoi256/MSPL)][[Bibtex](https://github.com/hchoi256/mspl#citation)]



## Updates

⛽⛽⛽ Contact: [eric970412@gmail.com](mailto:eric970412@gmail.com) or [hchoi256@kaist.ac.kr](mailto:hchoi256@kaist.ac.kr)

* **[2026.06.24]** 📄 Our ECCV-version paper is now available on [arXiv](https://arxiv.org/abs/2510.14792v4).
* **[2026.06.17]** 🎉 MSPL has been accepted to [ECCV 2026](https://eccv.ecva.net/).
* **[2026.04.04]** 🎉 MSPL has been accepted to the non-archival [CVPR 2026 Workshop MUSI](https://musi-workshop.github.io/).
* **[2026.03.26]** The code has been released and will continue to be updated.


## Installation

This project builds upon [MMDetection 3.x](https://github.com/open-mmlab/mmdetection/tree/3.x) and [BARON](https://github.com/wusize/ovdet).

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

bash proprocess.sh
```

## Quick Start
### Obtain CLIP Checkpoints
We use CLIP's ViT-B-16 model for the implementation of our method.
`pip install git+https://github.com/openai/CLIP.git` and run 
```python
import clip
import torch
model, _ = clip.load("ViT-B/16")
torch.save(model.state_dict(), 'checkpoints/clip_vitb16.pth')
```

### Offline Multi-Step Pseudo-Labeling

For pseudo-label generation,
- follow [the instructions](pseudo-labels/README.md) to generate pseudo-labels for base-category training images.
- or directly download the pre-generated pseudo-labels, [instances_train2017_pseudo_v0_new_caption.json](https://huggingface.co/datasets/hchoi256/MSPL/tree/main), from Hugging Face.

Note that OV-COCO and OV-LVIS share the same 118,287 training images, but their annotations differ.


### Training and Testing

The training and testing are supported [here](configs/baron/ov_coco/README.md).

## License

This project is released under the [NTU S-Lab License 1.0](LICENSE).

## Citation

```bibtex
@misc{choi2026msplmultisteppseudolabelingopenvocabulary,
      title={MSPL: Multi-Step Pseudo-Labeling for Open-Vocabulary Object Detection}, 
      author={Hojun Choi and Youngsun Lim and Jaeyo Shin and Hyunjung Shim},
      year={2026},
      eprint={2510.14792},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14792v4}, 
}
```

