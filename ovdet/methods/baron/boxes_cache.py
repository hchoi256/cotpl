import torch
import json
import torch.nn as nn
from mmcv.ops import nms
from mmdet.structures.bbox import scale_boxes, bbox_flip    # todo consider crop
from mmengine.structures import InstanceData
import torch.distributed as dist
from collections import defaultdict

base_class_indices = [1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 
                    21, 22, 23, 24, 26, 28, 29, 30, 33, 
                    37, 39, 42, 44, 45, 46, 47, 48, 49, 
                    50, 51, 53, 54, 56, 59, 61, 62, 63, 
                    64, 65, 68, 69, 70, 72, 73, 74, 75, 79]

class BoxesCache(nn.Module):
    def __init__(self, json_path, start_iter=10000,
                 num_proposals=300, nms_thr=0.1, score_thr=0.85, save=False):
        super(BoxesCache, self).__init__()
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            images_info = json_data['images']
            annotations_info = json_data['annotations']
        num_images = len(images_info)
        self.image_id2ordered_id = {info['id']: ordered_id for ordered_id, info in enumerate(images_info)}
        boxes = torch.zeros(num_images, num_proposals, 5)   # [x1, y1, x2, y2, s]

        # ========================
        # 1. box cache
        # ========================
        self.register_buffer("boxes", boxes, persistent=save)
        self.num_proposals = num_proposals
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.register_buffer("iter", torch.tensor(0, dtype=torch.long),
                             persistent=False)
        self.start_iter = start_iter

        # =======================
        # 2. anchor cache
        # =======================
        image2anchors = defaultdict(list)
        for ann in annotations_info:
            cat_id = ann['category_id']
            if cat_id in base_class_indices:
                continue
            img_id = ann['image_id']
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            image2anchors[img_id].append([x1, y1, x2, y2])

        max_anchors = max(len(v) for v in image2anchors.values()) if image2anchors else 1
        anchors_tensor = torch.zeros((num_images, max_anchors, 4))

        ordered_ids = [info['id'] for info in images_info]
        for i, img_id in enumerate(ordered_ids):
            anchors = image2anchors.get(img_id, [])
            if anchors:
                tensor = torch.tensor(anchors, dtype=torch.float32)
                anchors_tensor[i, :tensor.size(0)] = tensor

        # Register as buffers
        self.register_buffer("anchors", anchors_tensor, persistent=save)

    def forward(self, proposals):
        return self.update(proposals)

    @torch.no_grad()
    def update(self, proposals, ):
        self.iter.data += 1
        if self.iter < self.start_iter:
            return proposals
        nms_thr = self.nms_thr
        # TODO: pull cached boxes from all devices
        image_id = proposals.img_id

        bboxes = proposals.bboxes
        scores = proposals.scores

        scale_factor = proposals.scale_factor
        flip = proposals.flip
        flip_direction = proposals.flip_direction
        img_shape = proposals.img_shape

        if image_id not in self.image_id2ordered_id:
            return proposals
        ordered_id = self.image_id2ordered_id[image_id]
        cached_bboxes = self.boxes[ordered_id]
        cached_scores = cached_bboxes[:, -1]

        scaled_cached_bboxes = scale_boxes(cached_bboxes[:, :4], scale_factor)
        flipped_cached_bboxes = bbox_flip(scaled_cached_bboxes, img_shape, flip_direction) \
            if flip else scaled_cached_bboxes

        merged_boxes = torch.cat([flipped_cached_bboxes, bboxes])
        merged_scores = torch.cat([cached_scores, scores])

        score_kept = merged_scores > self.score_thr
        if score_kept.sum() == 0:
            score_kept = merged_scores.argmax().view(1)

        merged_boxes = merged_boxes[score_kept]
        merged_scores = merged_scores[score_kept]

        _, nmsed_kept = nms(merged_boxes, merged_scores, nms_thr)

        kept_boxes = merged_boxes[nmsed_kept]
        kept_scores = merged_scores[nmsed_kept]

        out = InstanceData(bboxes=kept_boxes,
                           scores=kept_scores,
                           metainfo=proposals.metainfo)

        # TODO: transform to the original size
        flipped_bboxes = bbox_flip(bboxes, img_shape, flip_direction) \
            if flip else bboxes
        scaled_back_bboxes = scale_boxes(flipped_bboxes, [1 / s for s in scale_factor])
        merged_boxes = torch.cat([cached_bboxes[:, :4], scaled_back_bboxes])
        merged_scores = torch.cat([cached_scores, scores])
        score_kept = merged_scores > self.score_thr
        if score_kept.sum() == 0:
            score_kept = merged_scores.argmax().view(1)

        merged_boxes = merged_boxes[score_kept]
        merged_scores = merged_scores[score_kept]

        _, nmsed_kept = nms(merged_boxes, merged_scores, nms_thr)

        kept_boxes = merged_boxes[nmsed_kept]
        kept_scores = merged_scores[nmsed_kept]

        num_update = min(self.num_proposals, len(kept_boxes))
        device = kept_scores.device
        update_cache_to_sync = torch.zeros(self.num_proposals, 6, device=device)     # [x,y,x,y,s,order_id]
        update_cache_to_sync[:, -1] = float(ordered_id)    # ordered_id
        update_cache = torch.cat([kept_boxes, kept_scores[:, None]], dim=1)[:num_update]
        update_cache_to_sync[:num_update, :5] = update_cache

        # sync for updates from other devices
        update_cache = self.sync_multiple_gpus(update_cache_to_sync)
        for update_cache_ in update_cache:
            ordered_id_ = int(update_cache_[0, -1].item())
            self.boxes[ordered_id_] = update_cache_[:, :5].to(device)  # update

        return out
    
    def get_anchors(self, image_id):
        if image_id not in self.image_id2ordered_id:
            raise ValueError(f"Image ID {image_id} not in anchor cache.")

        ordered_id = self.image_id2ordered_id[image_id]
        anchors = self.anchors[ordered_id]  # shape: (max_anchors, 4)
        valid_mask = (anchors.abs().sum(dim=1) > 0)

        return anchors[valid_mask].cpu().numpy()  # shape: (N, 4)


    @staticmethod
    def sync_multiple_gpus(tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        if get_world_size() == 1:
            return [tensor]
        with torch.no_grad():
            tensors_gather = [
                torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        return tensors_gather


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

if __name__ == "__main__":
    import os
    from random import choice
    from PIL import Image, ImageDraw

    # 1. 캐시 인스턴스 생성
    cache = BoxesCache(json_path='data/coco/hchoi/instances_train2017_pseudo.json')

    # 2. 임의의 image_id 하나 뽑기
    all_ids = list(cache.image_id2ordered_id.keys())
    while True:
        random_id = choice(all_ids)

        # 3. anchor 추출
        anchors = cache.get_anchors(random_id)

        # 4. 이미지 로드
        image_dir = 'data/coco/train2017'  # 이미지 디렉토리 경로 (필요 시 수정)
        image_filename = f'{random_id:012d}.jpg'
        image_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"[❌] Image not found at: {image_path}")

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # 5. anchor 시각화
        for box in anchors:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

        # 6. 결과 출력 및 저장
        print(f"\n[🔍] Random Image ID: {random_id}")
        print(f"[📦] Number of semantic anchors: {anchors.shape[0]}")
        print(f"[📋] Anchors:\n{anchors}")
        
        save_path = 'test.jpg'
        image.save(save_path)
        print(f"[💾] Saved visualized image to {save_path}")
        input()
