# Open-vocabulary COCO
## Data preparation

Prepare data following [MMDetection](https://github.com/open-mmlab/mmdetection). 
Obtain the json files for OV-COCO from [GoogleDrive](https://drive.google.com/drive/folders/1O6rt6WN2ePPg6j-wVgF89T7ql2HiuRIG?usp=sharing) and [huggingface](https://huggingface.co/datasets/hchoi256/CoT-PL/tree/main). Then, put them under `data/coco/hchoi`
The data structure looks like:

```text
checkpoints/
├── clip_vitb16.pth
├── res50_fpn_soco_star_400.pth
data/
├── coco
│   ├── annotations
│   │   ├── instances_{train,val}2017.json
│   ├── hchoi
│   │   ├── instances_train2017_base.json
│   │   ├── instances_train2017_pseudo_v0_new.json
│   │   ├── instances_val2017_base.json
│   │   ├── instances_val2017_novel.json
│   │   ├── captions_train2017_tags_allcaps.json
│   ├── train2017
│   ├── val2017
│   ├── test2017
```

Otherwise, generate the json files using the following scripts
```bash
python tools/pre_processors/keep_coco_base.py \
      --json_path data/coco/annotations/instances_train2017.json \
      --out_path data/coco/hchoi/instances_train2017_base.json
```
```bash
python tools/pre_processors/keep_coco_base.py \
      --json_path data/coco/annotations/instances_val2017.json \
      --out_path data/coco/hchoi/instances_val2017_base.json
```
```bash
python tools/pre_processors/keep_coco_novel.py \
      --json_path data/coco/annotations/instances_val2017.json \
      --out_path data/coco/hchoi/instances_val2017_novel.json
```
The json file for caption supervision `captions_train2017_tags_allcaps.json` is obtained following 
[Detic](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md#:~:text=Next%2C%20we%20preprocess%20the%20COCO%20caption%20data%3A). Put it under 
`data/coco/hchoi`.

```bash
bash reprod/preprocess.sh
```
Lastly, apply the pseudo-label categories into the framework.

The pseudo-label generation is on [pseudo-label](pseudo-labels/README.md) or download it directly from [huggingface](https://huggingface.co/datasets/hchoi256/CoT-PL/tree/main).

### Class Embeddings
As the training on COCO tends to converge to base categories, we use the output of the last attention
layer for classification. Generate the class embeddings by 
```bash
python tools/hand_craft_prompt.py --model_version ViT-B/16 --ann data/coco/annotations/instances_val2017.json \
--out_path data/metadata/coco_clip_hand_craft.npy --dataset coco
```
```bash
python tools/hand_craft_prompt.py --model_version ViT-B/16 --ann data/coco/hchoi/instances_train2017_pseudo_v0_new.json \
--out_path data/metadata/coco_clip_hand_craft_pseudo_v0_new.npy --dataset coco
```
```bash
python tools/hand_craft_prompt.py --out_path coco_clip_hand_craft_background.npy --background
```
The generated files are used for training and testing, respectively.


## Testing
### Open Vocabulary COCO
The implementation based on MMDet3.x achieves better results compared to the results reported in the paper.

|             | Backbone |  Method  | Supervision  | Novel AP50 |                                        Config                                        |         Download          |
|:-----------:|:--------:|:--------:|:------------:|:----------:|:------------------------------------------------------------------------------------:|:-------------------------:|
|  Paper  | R-50-FPN |  BARON   |     CLIP     |    34.0    |    -     | - |
|  This Repo  | R-50-FPN |  BARON   |     CLIP     |    34.6    |    [config](baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py)     | [model](https://drive.google.com/drive/folders/1JTM0uoPQZtq7lnhZxCBwjxBUca9omYR9?usp=sharing) &#124;  [log](https://drive.google.com/drive/folders/1JTM0uoPQZtq7lnhZxCBwjxBUca9omYR9?usp=sharing) |
|  Paper  | R-50-FPN |  CoT-PL   |     CLIP     |    41.7    |    -     | - |
|  This Repo  | R-50-FPN |  CoT-PL   |     CLIP     |    42.2    |    [config](baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py)     | [model](https://drive.google.com/drive/folders/1d06y8DxfgkitPRGuk3HSp8BwG7ZFRLeu?usp=sharing) &#124;  [log](https://drive.google.com/drive/folders/1d06y8DxfgkitPRGuk3HSp8BwG7ZFRLeu?usp=sharing) |

To test the models, run
```bash
bash tools/dist_test.sh configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py path/to/save/logs/and/checkpoints <NUM_GPUS> <GPU_IDS>
# bash tools/dist_test.sh configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py checkpoints/iter_90000.pth 2 6,7
```

## Training
### Contrastive Background Learning for CLIP Knowledge Distillation
Train the detector based on FasterRCNN+ResNet50+FPN with SyncBN and SOCO pre-trained model. Obtain the SOCO pre-trained 
model from [GoogleDrive](https://drive.google.com/file/d/1rIW9IXjWEnFZa4klZuZ5WNSchRYaOC0x/view?usp=sharing) and put it
under `checkpoints`.
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PORT=<PORT> bash tools/dist_train.sh configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py <NUM_GPUS> <GPU_IDS> <SEED> --work-dir work_dirs/
# CUBLAS_WORKSPACE_CONFIG=:4096:8 PORT=39000 bash tools/dist_train.sh configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py 2 6,7 1194806617 --work-dir work_dirs/
```