import json
import os

# Merge all batch output files into a single annotation file
folder_path = '../pseudo_v0'

merged_data = {}
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.update(data)

with open('result/raw_psuedo_labels.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)