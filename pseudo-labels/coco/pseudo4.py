import json
from tqdm import tqdm

# Convert the pseudo-labels to match the COCO format
pseudo_classes = ['dog', 'knife', 'fish', 'stick', 'cup', 'elephant', 'box', 'cat', 'lamp', 'airplane', 'house', 'umbrella', 'pen', 'camera', 'desk', 'plate', 'table', 'door', 'gun', 'cell phone', 'cow', 'cake', 'plane', 'shoe', 'skateboard', 'phone', 'bus', 'light', 'wine glass', 'cabinet', 'traffic light', 'cloth', 'keyboard', 'window', 'wall', 'bone', 'hand', 'sword', 'triangle', 'worm', 'bridge', 'shirt', 'pillow', 'stone', 'square', 'fruit', 'tennis racket', 'computer', 'ship', 'snake', 'fire hydrant', 'sink', 'bread', 'stop sign', 'tomato', 'couch', 'arm', 'basket', 'bathroom', 'bat', 'tennis player', 'leg', 'chicken', 'sign']
top_categories = set(pseudo_classes)

with open('result/unseen_psuedo_labels.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for img_id, img_data in tqdm(data.items()):
    fg = img_data.get('foreground', {})
    fg_bboxes = fg.get('bbox', [])
    fg_categories = fg.get('category', [])

    new_fg_bboxes = []
    new_fg_categories = []

    for bbox, category in zip(fg_bboxes, fg_categories):
        if category in top_categories:
            new_fg_bboxes.append(bbox)
            new_fg_categories.append(category)

    img_data['foreground']['bbox'] = new_fg_bboxes
    img_data['foreground']['category'] = new_fg_categories

with open('result/topk_unseen_psuedo_labels.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
