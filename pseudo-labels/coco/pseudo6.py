# Print the class weights for base classes combined with pseudo-labeled classes
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

pseudo_classes = ['dog', 'knife', 'fish', 'stick', 'cup', 'elephant', 'box', 'cat', 'lamp', 'airplane', 'house', 'umbrella', 'pen', 'camera', 'desk', 'plate', 'table', 'door', 'gun', 'cell phone', 'cow', 'cake', 'plane', 'shoe', 'skateboard', 'phone', 'bus', 'light', 'wine glass', 'cabinet', 'traffic light', 'cloth', 'keyboard', 'window', 'wall', 'bone', 'hand', 'sword', 'triangle', 'worm', 'bridge', 'shirt', 'pillow', 'stone', 'square', 'fruit', 'tennis racket', 'computer', 'ship', 'snake', 'fire hydrant', 'sink', 'bread', 'stop sign', 'tomato', 'couch', 'arm', 'basket', 'bathroom', 'bat', 'tennis player', 'leg', 'chicken', 'sign']

not_in_coco = [cls for cls in pseudo_classes if cls not in coco_classes]
print("Extra COCO classes:", not_in_coco)

presence_list = [1 if cls in pseudo_classes else 0 for cls in coco_classes]

base_class_weight = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                    0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                    0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                    1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                    0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                    1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
                    0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                    1, 0, 1, 1, 1, 1, 0, 0, 0, 1]

combined_class_weight = [int(a or b) for a, b in zip(base_class_weight, presence_list)]

print("Class weights:", combined_class_weight)
