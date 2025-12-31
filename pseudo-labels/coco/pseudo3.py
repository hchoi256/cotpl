import json
from collections import defaultdict

# Select pseudo-labels whose confidence exceeds a predefined threshold
with open('result/unseen_psuedo_labels.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

bbox_count = defaultdict(int)
total_annotations = 0

for img_id, img_data in data.items():
    categories = img_data.get('foreground', {}).get('category', [])
    for category in categories:
        bbox_count[category] += 1
        total_annotations += 1

threshold = 1294 # Minimum number of annotations required for a base class
filtered_classes = [(cat, cnt) for cat, cnt in bbox_count.items() if cnt >= threshold]

filtered_classes = sorted(filtered_classes, key=lambda x: x[1], reverse=True)

# for category, count in filtered_classes:
#     print(f"{category}: {count} bboxes")

pseudo_classes = [category for category, _ in filtered_classes]
print(pseudo_classes)

# print(f"\nTotal number of annotations: {total_annotations}")