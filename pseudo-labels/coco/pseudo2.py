import json

# Extract classes that are not part of the base class set
base_class = ["toilet", "bicycle", "apple", "train", "laptop", "carrot", "motorcycle", "oven", "chair",
    "mouse", "boat", "kite", "sheep", "horse", "sandwich", "clock", "tv", "backpack", "toaster",
    "bowl", "microwave", "bench", "book", "orange", "bird", "pizza", "fork", "frisbee", "bear",
    "vase", "toothbrush", "spoon", "giraffe", "handbag", "broccoli", "refrigerator", "remote",
    "surfboard", "car", "bed", "banana", "donut", "skis", "person", "truck", "bottle", "suitcase", "zebra"]

exclude_keywords = ["tree", "flower", "sky", "mountain", "plant", "leaf", "wood", "letter", "rock",
                    "baseball player", "building", "man", "grass", "branch", "leaves"]

with open('result/raw_psuedo_labels.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for img_id, img_data in data.items():
    fg = img_data.get('foreground', {})
    fg_bboxes = fg.get('bbox', [])
    fg_categories = fg.get('category', [])

    new_fg_bboxes = []
    new_fg_categories = []
    moved_bboxes_to_bg = []

    for bbox, category in zip(fg_bboxes, fg_categories):
        if category in base_class or any(keyword in category for keyword in exclude_keywords):
            moved_bboxes_to_bg.append(bbox)
        else:
            new_fg_bboxes.append(bbox)
            new_fg_categories.append(category)

    img_data['foreground']['bbox'] = new_fg_bboxes
    img_data['foreground']['category'] = new_fg_categories

    if 'background' not in img_data:
        img_data['background'] = {'bbox': []}
    if 'bbox' not in img_data['background']:
        img_data['background']['bbox'] = []

    img_data['background']['bbox'].extend(moved_bboxes_to_bg)

with open('result/unseen_psuedo_labels.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)