import json
from tqdm import tqdm

# Merge the pseudo-labels with the base dataset
coco_file = '../../data/coco/pseudo/instances_train2017_base.json'
pseudo_label_file = 'result/topk_unseen_psuedo_labels.json'
output_file = 'result/instances_train2017_pseudo.json'

pseudo_classes = ['dog', 'knife', 'fish', 'stick', 'cup', 'elephant', 'box', 'cat', 'lamp', 'airplane', 'house', 'umbrella', 'pen', 'camera', 'desk', 'plate', 'table', 'door', 'gun', 'cell phone', 'cow', 'cake', 'plane', 'shoe', 'skateboard', 'phone', 'bus', 'light', 'wine glass', 'cabinet', 'traffic light', 'cloth', 'keyboard', 'window', 'wall', 'bone', 'hand', 'sword', 'triangle', 'worm', 'bridge', 'shirt', 'pillow', 'stone', 'square', 'fruit', 'tennis racket', 'computer', 'ship', 'snake', 'fire hydrant', 'sink', 'bread', 'stop sign', 'tomato', 'couch', 'arm', 'basket', 'bathroom', 'bat', 'tennis player', 'leg', 'chicken', 'sign']
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

with open(pseudo_label_file, 'r') as f:
    pseudo_data = json.load(f)

existing_category_map = {cat['name']: cat['id'] for cat in coco_data['categories']}

last_category_id = max(cat['id'] for cat in coco_data['categories'])
new_category_map = {}

for class_name in pseudo_classes:
    if class_name in existing_category_map:
        new_category_map[class_name] = existing_category_map[class_name]
    else:
        last_category_id += 1
        coco_data['categories'].append({
            'id': last_category_id,
            'name': class_name,
            'supercategory': 'pseudo'
        })
        new_category_map[class_name] = last_category_id

annotation_id = 909000554145 + 1  # max id + 1

for image_id_str, info in tqdm(pseudo_data.items()):
    image_id = int(image_id_str)

    fg_bboxes = info.get('foreground', {}).get('bbox', [])
    fg_categories = info.get('foreground', {}).get('category', [])

    for bbox, category_name in zip(fg_bboxes, fg_categories):
        area = bbox[2] * bbox[3]

        category_id = new_category_map.get(category_name)
        if category_id is None:
            print(f"Warning: category '{category_name}' not found in new_category_map.")
            continue

        coco_data['annotations'].append({
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'area': area,
            'iscrowd': 0,
        })

        annotation_id += 1

with open(output_file, 'w') as f:
    json.dump(coco_data, f, indent=2)
