import os
import json
from PIL import Image
import numpy as np
import torch
import cv2
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from qwen import Qwen

# Initialize Qwen and SAM
qwen = Qwen()

def init_sam(gpu_id=0):
    torch.cuda.set_device(gpu_id)
    sam_ckpt_path = "../checkpoints/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to(f'cuda:{gpu_id}')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    return mask_generator

# Mask postprocessing
def filter(keep: torch.Tensor, masks_result) -> list:
    keep = keep.int().cpu().numpy()
    return [masks_result[i] for i in keep]

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2):
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks, num_masks), dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks, num_masks), dtype=torch.float, device=masks.device)

    for i in range(num_masks):
        for j in range(i, num_masks):
            inter = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = inter / union
            iou_matrix[i, j] = iou
            if inter / masks_area[i] < 0.5 and inter / masks_area[j] >= 0.85:
                inner = 1 - (inter / masks_area[j]) * (inter / masks_area[i])
                inner_iou_matrix[i, j] = inner
            if inter / masks_area[i] >= 0.85 and inter / masks_area[j] < 0.5:
                inner = 1 - (inter / masks_area[j]) * (inter / masks_area[i])
                inner_iou_matrix[j, i] = inner

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_max_u, _ = torch.triu(inner_iou_matrix, diagonal=1).max(dim=0)
    inner_max_l, _ = torch.tril(inner_iou_matrix, diagonal=1).max(dim=0)

    keep = (iou_max <= iou_thr) & (scores > score_thr) & \
           (inner_max_u <= 1 - inner_thr) & (inner_max_l <= 1 - inner_thr)

    if keep.sum() == 0:
        keep[scores.topk(3).indices] = True

    return idx[keep]

def masks_update(masks, iou_thr=0.8, score_thr=0.7, inner_thr=0.5):
    if len(masks) == 0:
        return []

    seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks])).to('cuda')
    iou_pred = torch.from_numpy(np.array([m['predicted_iou'] for m in masks])).to('cuda')
    stability = torch.from_numpy(np.array([m['stability_score'] for m in masks])).to('cuda')

    scores = stability * iou_pred
    keep_idx = mask_nms(seg_pred, scores, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr)
    return filter(keep_idx, masks)


# Core image processor
def process_single_image(img_file, data_dir, mask_generator):
    try:
        img_path = os.path.join(data_dir, img_file)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks_raw = mask_generator.generate(image_rgb)
        masks = masks_update(masks_raw)

        img_id = str(int(os.path.splitext(img_file)[0]))

        foreground_bboxes = []
        foreground_categories = []
        background_bboxes = []
        background_categories = []

        for idx, mask in enumerate(masks):
            segmentation = mask['segmentation']
            if np.sum(segmentation) == 0:
                continue
                
            bbox = mask['bbox']
            x, y, w, h = map(int, bbox)
            if w * h < 10000: # filter out small masks
                continue
            
            # Image preprocessing: blurring & grayscale
            ## apply black mask
            masked_img = np.zeros_like(image)

            ## apply grayscale + blur
            bbox_crop = image[y:y+h, x:x+w]
            bbox_gray = cv2.cvtColor(bbox_crop, cv2.COLOR_BGR2GRAY)
            bbox_gray_blur = cv2.GaussianBlur(bbox_gray, (31, 31), 0)
            bbox_gray_blur_rgb = cv2.cvtColor(bbox_gray_blur, cv2.COLOR_GRAY2BGR)
            masked_img[y:y+h, x:x+w] = bbox_gray_blur_rgb

            ## apply saturation
            rgb_patch = image.copy()
            hsv = cv2.cvtColor(rgb_patch, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * 1.2, 0, 255)
            rgb_emphasized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            masked_img[segmentation] = rgb_emphasized[segmentation]

            ## apply bounding box
            vis_img = masked_img.copy()
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))

            # query mllm
            result, category = qwen.forward(image_pil)

            if result.lower() == "yes":
                foreground_bboxes.append(bbox)
                foreground_categories.append(category)
            elif result.lower() == "no":
                background_bboxes.append(bbox)
                background_categories.append(category)


        return img_id, {
            "foreground": {
                "bbox": foreground_bboxes,
                "category": foreground_categories
            },
            "background": {
                "bbox": background_bboxes,
                "category": background_categories
            }
        }

    except Exception as e:
        print(f"[ERROR] {img_file}: {e}")
        return None

# Dataset processing
def process_dataset(data_dir, output_path, start_idx=0, end_idx=None):
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
    if end_idx is None:
        end_idx = len(image_files)

    image_files = image_files[start_idx:end_idx]
    results = []

    mask_generator = init_sam(gpu_id=0)

    for img_file in tqdm(image_files, desc="Processing"):
        result = process_single_image(img_file, data_dir, mask_generator)
        if result is not None:
            results.append(result)

    annotations = {img_id: bboxes_dict for (img_id, bboxes_dict) in results}

    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nSaved annotations for {len(annotations)} images to {output_path}")


# CLI
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/coco/train2017')
    parser.add_argument('--output_path', type=str, default='0.json')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()

    process_dataset(
        data_dir=args.data_dir,
        output_path=args.output_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
