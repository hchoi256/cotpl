# Pseudo-Label Generation Process
Pseudo-labels for unlabeled COCO images are generated through a two-phase workflow: (1) pseudo-label generation and (2) pseudo-label post-processing.

## Installation

Install required packages as follows:
run
```bash
cd pseudo-labels/
pip install transformers
pip install qwen-vl-utils

cd segment-anything; pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

Place the checkpoint `sam_vit_h_4b8939.pth` in the `checkpoints/` directory.

## Pseudo-Label Generation
In our setting, pseudo-annotations are generated for a subset of 10,000 training images using a single GPU.

The process can be parallelized across 12 GPUs to handle the full 118,287 images in the COCO training set.

```bash
CUDA_VISIBLE_DEVICES=0 python hand_craft_sam.py --start 0 --end 10000 --output_path result/10000.json
CUDA_VISIBLE_DEVICES=1 python hand_craft_sam.py --start 10000 --end 20000 --output_path result/20000.json
...
CUDA_VISIBLE_DEVICES=11 python hand_craft_sam.py --start 110000 --end 118287 --output_path result/118287.json
```

By default, the proposed pipeline employs the MLLM `Qwen2-VL-7B-Instruct`.
- Optionally, if you want to change a different model, such as InstructBLIP, replace the line `model = Qwen()` with the other model in the `hand_craft_sam.py` file.

## Pseudo-Label Post-Processing
Place the resulting batched pseudo-labels in the `pseudo_v0_new/` directory (or download them from `pseudo-labels.zip` on [huggingface](https://huggingface.co/datasets/hchoi256/CoT-PL/tree/main)) and then navigate to the `coco/` directory, which contains post-processing scripts such as `pseudo<Number>.py`.

Additionally, create a `result/` directory to store the processed outputs as follows:

```text
coco/
├── pseudo1.py
├── pseudo2.py
├── pseudo3.py
├── pseudo4.py
├── pseudo5.py
├── pseudo6.py
├── result/
pseudo_v0_new/
├── 10000.json
├── 20000.json
├── 30000.json
...
├── 118287.json
```

We provide a default setup for pseudo-label generation. Run the following:

```bash
cd ../coco
bash run.sh
```

This automation follows the procedure below:

<details>
  <summary>Here</summary>

```bash
cd coco/
python pseudo1.py # Merge all batch output files
python pseudo2.py # Extract unseen classes
python pseudo3.py # Select pseudo-labels that exceeds a threshold

Expected output:
['dog', 'knife', 'fish', 'stick', 'cup', 'elephant', 'box', 'pole', 'cat', 'lamp', 'airplane', 'house', 'umbrella', 'pen', 'camera', 'desk', 'plate', 'table', 'door', 'gun', 'cell phone', 'cow', 'cake', 'plane', 'shoe', 'skateboard', 'phone', 'bus', 'light', 'wine glass', 'cabinet', 'traffic light', 'cloth', 'keyboard', 'window', 'wall', 'bone', 'hand', 'sword', 'triangle', 'worm', 'bridge', 'shirt', 'pillow', 'stone', 'square', 'fruit', 'tennis racket', 'computer', 'ship', 'snake', 'fire hydrant', 'sink', 'bread', 'stop sign', 'tomato', 'couch', 'arm', 'basket', 'bathroom', 'bat', 'tennis player', 'leg', 'chicken', 'sign']
```

Copy and paste the output from `pseudo3.py` into the `pseudo_classes` variable in the following files: `pseudo4.py`, `pseudo5.py`, and `pseudo6.py`.

```bash
python pseudo4.py // Match the COCO format
python pseudo5.py // Merge the pseudo-labels with the base dataset
python pseudo6.py // Print the class weights

Expected output:
Extra COCO classes: ['fish', 'stick', 'box', 'lamp', 'house', 'pen', 'camera', 'desk', 'plate', 'table', 'door', 'gun', 'plane', 'shoe', 'phone', 'light', 'cabinet', 'cloth', 'window', 'wall', 'bone', 'hand', 'sword', 'triangle', 'worm', 'bridge', 'shirt', 'pillow', 'stone', 'square', 'fruit', 'computer', 'ship', 'snake', 'bread', 'tomato', 'arm', 'basket', 'bathroom', 'bat', 'tennis player', 'leg', 'chicken', 'sign'] // the number of classes is 44
Class weights: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
```

Update the class weights in the configuration file. Set the `num_pls` variable to the number of extra COCO classes, as shown below:


```text
// inside the baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py file
num_pls = 44
class_weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1] + [1] * num_pls + [0.7]
```

Copy and paste the extra COCO classes into the `metainfo` section of the `coco.py` file as shown below:


```text
# inside the `../reprod/coco.py` file
METAINFO = {'classes': ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
# pseudo-labels
'fish', 'stick', 'box', 'lamp', 'house', 'pen', 'camera', 'desk', 'plate', 'table', 'door', 'gun', 'plane', 'shoe', 'phone', 'light', 'cabinet', 'cloth', 'window', 'wall', 'bone', 'hand', 'sword', 'triangle', 'worm', 'bridge', 'shirt', 'pillow', 'stone', 'square', 'fruit', 'computer', 'ship', 'snake', 'bread', 'tomato', 'arm', 'basket', 'bathroom', 'bat', 'tennis player', 'leg', 'chicken', 'sign'),
```

</details>

Place the resulting file (`result/instances_train2017_pseudo.json`) in the COCO (`data/coco/hchoi/`) directory for the use of the generated pseudo-labels.

Lastly, update the `coco.py` file as follows:

```bash
cd ../reprod/
bash proprocess.sh
```