import math
import random
import torch
import numpy as np
from ovdet.utils import multi_apply


def get_normed_boxes(boxes, spanned_box):
    spanned_box_shape = spanned_box[2:] - spanned_box[:2]
    boxes = boxes.reshape(-1, 2, 2) - spanned_box[:2].reshape(1, 1, 2)
    boxes = boxes / (spanned_box_shape.reshape(1, 1, 2) + 1e-12)

    return boxes.reshape(-1, 4)


def get_spanned_box(boxes, image_size=None):
    # boxes x, y
    # image_size h, w
    corner_points = boxes.reshape(-1, 2)
    bottom_right = corner_points.max(0)
    upper_left = corner_points.min(0)

    upper_left_bottom_right = np.stack([upper_left, bottom_right], axis=0)
    if image_size is not None:
        upper_left_bottom_right = clamp_with_image_size(upper_left_bottom_right, image_size)

    return upper_left_bottom_right.reshape(4)


def clamp_with_image_size(points, image_size):
    points[:, 0] = np.clip(points[:, 0], a_min=0, a_max=image_size[1])
    points[:, 1] = np.clip(points[:, 1], a_min=0, a_max=image_size[0])

    return points


def perm_generator(seq):
    seen = set()
    length = len(seq)
    while True:
        perm = tuple(random.sample(seq, length))
        if perm not in seen:
            seen.add(perm)
            yield perm


def pseudo_permutations(seq_length, num_permutation):
    rand_perms = perm_generator(list(range(seq_length)))
    return [list(next(rand_perms)) for _ in range(num_permutation)]


class NeighborhoodSampling:
    """
        checkboard:   0  1  2
                      3  4  5
                      6  7  8
        context boxes: [0, 1, 2, 3, 5, 6, 7, 8]
        candidate_groups: 2 ** 8 = 256
        box: tensor
    """
    def __init__(self,
                 max_groups=4,
                 max_permutations=4,
                 alpha=3.0,
                 cut_off_thr=0.5,
                 base_probability=0.5,
                 interval=0.0,
                 use_anchor=False,
                 **kwargs):
        self.interval = interval
        box_ids = []
        left_right_up_downs = []
        box_templates = []
        for i in range(3):
            h_interval = (float(i) - 1.0) * self.interval
            for j in range(3):
                w_interval = (float(j) - 1.0) * self.interval
                box = [float(j) + w_interval, float(i) + h_interval,
                       float(j+1) + w_interval, float(i+1) + h_interval]
                box_templates.append(box)
        self.box_templates = np.array(box_templates, dtype=np.float32)
        self.binary_mask_template = 10 ** np.arange(9, dtype=np.float32)


        for l in range(2):       # left: -1
            for r in range(2):   # right +1
                for u in range(2):    # up -3
                    for d in range(2):  # down  +3
                        left_right_up_downs.append([l, r, u, d])
                        box_ids.append(list({4-l-3*u, 4-3*u, 4+r-3*u,
                                             4-l,     4,     4+r,
                                             4-l+3*d, 4+3*d, 4+r+3*d}))
        self.box_ids = box_ids
        self.alpha = alpha
        self.cut_off_thr = cut_off_thr
        self.left_right_up_downs = np.array(left_right_up_downs, dtype=np.float32)
        self.max_groups = max_groups
        self.max_permutations = max_permutations
        self.base_probability = base_probability
        self.context_box_ids = [0, 1, 2, 3, 5, 6, 7, 8]
        self.use_anchor = use_anchor
        self.directions = {
            "top-left": 0,
            "top": 1,
            "top-right": 2,
            "left": 3,
            "center": 4,         
            "right": 5,
            "bottom-left": 6,
            "bottom": 7,
            "bottom-right": 8
        }
        self.anchors = []

    def put_anchors(self, anchors):
        self.anchors = anchors
        
    @staticmethod
    def _get_group_id(left_right_up_down):
        assert len(left_right_up_down) == 4
        # list of {0, 1}'s
        left, right, up, down = left_right_up_down
        return left * (2 ** 3) + right * (2 ** 2) + up * 2 + down

    def _get_left_right_up_down_possibility(self, box, image_size):
        img_h, img_w = image_size
        box_w, box_h = box[2] - box[0] + 1e-12, box[3] - box[1] + 1e-12
        box_h_w_ratio = box_h / box_w
        box_w_h_ratio = box_w / box_h
        # Initiate: <, >, ^, v
        left_right_up_down = (np.array([box_h_w_ratio, box_h_w_ratio,
                                        box_w_h_ratio, box_w_h_ratio],
                                       dtype=np.float32) ** self.alpha) * self.base_probability
        # check boundary
        boundary_check = np.array([box[0] / box_w, (img_w - box[2]) / box_w,
                                   box[1] / box_h, (img_h - box[3]) / box_h],
                                  dtype=np.float32) > (self.cut_off_thr + self.interval)
        left_right_up_down = left_right_up_down * boundary_check.astype(np.float32)
        left_right_up_down = np.clip(left_right_up_down, a_min=0.0, a_max=self.base_probability)
        left_right_up_down[np.isnan(left_right_up_down)] = 0.0

        return left_right_up_down

    def group_generator(self, box_possibilities):
        assert box_possibilities[4] == 1.0       # center box (roi) are fixed at 1.0
        seen = set()
        while True:
            try:
                sampled_mask = torch.bernoulli(torch.from_numpy(box_possibilities)).numpy()
            except:
                raise ValueError(f"Invalid box_possibilities{box_possibilities}")
            box_ids = sorted(sampled_mask.nonzero()[0].tolist())
            box_ids_str = ''.join([str(box_id) for box_id in box_ids])
            if box_ids_str not in seen:
                seen.add(box_ids_str)
                yield box_ids

    @staticmethod
    def _get_base_possibilities(left_right_up_down_possibility):
        box_possibilities = np.ones(9, dtype=np.float32)
        box_possibilities[[0, 3, 6]] *= left_right_up_down_possibility[0]
        box_possibilities[[2, 5, 8]] *= left_right_up_down_possibility[1]
        box_possibilities[[0, 1, 2]] *= left_right_up_down_possibility[2]
        box_possibilities[[6, 7, 8]] *= left_right_up_down_possibility[3]
        box_possibilities[[0, 2, 6, 8]] **= 0.5

        return box_possibilities
    
    def _get_anchor_box_indices(self, box):
        x1, y1, x2, y2 = box

        anchor_centers = np.stack([
            (self.anchors[:, 0] + self.anchors[:, 2]) / 2,
            (self.anchors[:, 1] + self.anchors[:, 3]) / 2
        ], axis=1)

        anchor_box_indices = np.full(len(self.anchors), -1, dtype=np.int32)

        for i, (cx, cy) in enumerate(anchor_centers):
            if cx < x1 and cy < y1:
                direction = "top-left"
            elif x1 <= cx <= x2 and cy < y1:
                direction = "top"
            elif cx > x2 and cy < y1:
                direction = "top-right"
            elif cx < x1 and y1 <= cy <= y2:
                direction = "left"
            elif x1 <= cx <= x2 and y1 <= cy <= y2:
                direction = "center"
            elif cx > x2 and y1 <= cy <= y2:
                direction = "right"
            elif cx < x1 and cy > y2:
                direction = "bottom-left"
            elif x1 <= cx <= x2 and cy > y2:
                direction = "bottom"
            elif cx > x2 and cy > y2:
                direction = "bottom-right"
            else:
                assert False, "Unexpected anchor position."

            anchor_box_indices[i] = self.directions[direction]

        box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        for direction in self.context_box_ids:
            matched_idxs = np.where(anchor_box_indices == direction)[0]
            if len(matched_idxs) <= 1:
                continue

            matched_centers = anchor_centers[matched_idxs]
            dists = np.sum((matched_centers - box_center) ** 2, axis=1)
            keep_idx = matched_idxs[np.argmin(dists)]

            for idx in matched_idxs:
                if idx != keep_idx:
                    anchor_box_indices[idx] = 4

        return anchor_box_indices
    
    def _get_box_templates(self, box):
        box_w, box_h = box[2] - box[0] + 1e-12, box[3] - box[1] + 1e-12
        box_templates = self.box_templates * np.array([box_w, box_h, box_w, box_h], dtype=np.float32).reshape(1, 4)
        
        center_box_template = box_templates[4]  # [1, 1, 2, 2]
        off_set = np.array(box, dtype=np.float32) - center_box_template
        box_templates = box_templates + off_set.reshape(1, 4)

        if self.use_anchor:
            anchor_box_indices = self._get_anchor_box_indices(box)
            for anchor, anchor_id in zip(self.anchors, anchor_box_indices):
                if anchor_id == 4:
                    continue
                boxes = np.array([box_templates[4], anchor])
                box_templates[anchor_id] = get_spanned_box(boxes)
        else:
            anchor_box_indices = None
        # assert self.use_anchor, "self.use_anchor is set to False!"
        return box_templates, anchor_box_indices
    
    def _get_box_possibilities(self, box, box_templates, anchor_box_indices, image_size):
        # center_box = box_templates[4]

        # base neighborhood probability
        left_right_up_down_possibility = self._get_left_right_up_down_possibility(box, image_size)
        box_possibilities = self._get_base_possibilities(left_right_up_down_possibility)

        # anchor-based probability
        # if self.use_anchor:
        #     anchor_directions = np.unique(anchor_box_indices)
        #     anchor_directions = anchor_directions[anchor_directions != 4]

        #     if len(anchor_directions) > 0:
        #         neighbor_boxes = box_templates[anchor_directions]                     # shape: [N, 4]
        #         center_boxes = np.repeat(center_box[None, :], len(anchor_directions), axis=0)  # [N, 4]

        #         stacked = np.stack([center_boxes, neighbor_boxes], axis=1)           # [N, 2, 4]
        #         spanned_boxes = np.array([get_spanned_box(pair, image_size=image_size) for pair in stacked])  # [N, 4]

        #         widths = spanned_boxes[:, 2] - spanned_boxes[:, 0] + 1e-12
        #         heights = spanned_boxes[:, 3] - spanned_boxes[:, 1] + 1e-12
        #         aspect_ratios = heights / widths

        #         probs = (aspect_ratios ** self.alpha) * self.base_probability
        #         probs = np.clip(probs, a_min=0.0, a_max=self.base_probability)

        #         for idx, i in enumerate(anchor_directions):
        #             if box_possibilities[i] < probs[idx]:
        #                 box_possibilities[i] = probs[idx]

        return box_possibilities


    def _get_box_ids_per_group(self, box, box_templates, anchor_box_indices, image_size):
        box_possibilities = self._get_box_possibilities(box, box_templates, anchor_box_indices, image_size)
        box_possibilities = np.clip(box_possibilities, a_min=0.0, a_max=1.0)

        num_valid_context_boxes = int((box_possibilities > 0.0).sum()) - 1
        num_groups = min(self.max_groups, math.factorial(num_valid_context_boxes))
        random_gen = self.group_generator(box_possibilities)
        box_ids_per_group = [next(random_gen) for _ in range(num_groups)]
        
        return box_ids_per_group
        
    def sample(self, box, image_size):
        # build open-world compositional structure
        box_templates, anchor_box_indices = self._get_box_templates(box)
        box_ids_per_group = self._get_box_ids_per_group(box, box_templates, anchor_box_indices, image_size)

        groups, normed_boxes, spanned_boxes, box_ids = multi_apply(self._sample_boxes_per_group,
                                                                   box_ids_per_group,
                                                                   image_size=image_size,
                                                                   box_templates=box_templates)
        box_ids = [box_id for box_ids_ in box_ids for box_id in box_ids_]

        return groups, normed_boxes, spanned_boxes, box_ids

    def _sample_boxes_per_group(self, box_ids, image_size, box_templates):   # TODO: paralell

        num_boxes = len(box_ids)

        # pseudo_perm = list(range(num_boxes))
        # all_permutations = [pseudo_perm, pseudo_perm[::-1]]
        boxes = box_templates[box_ids]
        boxes = clamp_with_image_size(boxes.reshape(-1, 2), image_size).reshape(-1, 4)
        spanned_box = get_spanned_box(boxes)
        normed_boxes = get_normed_boxes(boxes, spanned_box)

        num_permutations = min(math.factorial(num_boxes), self.max_permutations)
        # all_permutations = [list(perm) for perm in permutations(range(num_boxes))]
        # box_permutations = random.sample(all_permutations, k=num_permutations)

        box_permutations = pseudo_permutations(num_boxes, num_permutations)

        boxes_list = [torch.from_numpy(boxes[perm]) for perm in box_permutations]
        normed_boxes_list = [torch.from_numpy(normed_boxes[perm]) for perm in box_permutations]
        box_ids_list = [[box_ids[idx] for idx in perm] for perm in box_permutations]

        return boxes_list, normed_boxes_list, torch.from_numpy(spanned_box), box_ids_list

def draw_box_templates(image_size, box, box_templates, anchors=None, save_path='debug_templates/template1.png'):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os

    direction_labels = [
        "TL", "T", "TR",
        "L",  "C", "R",
        "BL", "B", "BR"
    ]

    h, w = image_size
    fig, ax = plt.subplots(1, figsize=(8, 10))

    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])
    ax.set_aspect('equal')
    ax.axis('off')

    # ROI (center box)
    x1, y1, x2, y2 = box
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor='red', facecolor='none'))
    ax.text(x1, y1 - 5, 'ROI', color='red', fontsize=8)

    # box_templates: T0~T8 => TL, T, TR, ...
    for i, tmpl in enumerate(box_templates):
        x1, y1, x2, y2 = tmpl
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=1, edgecolor='blue', facecolor='none', linestyle='--'))
        ax.text(x1, y1 - 3, f"{direction_labels[i]}", color='blue', fontsize=6)

    # ✅ anchor boxes
    if anchors is not None:
        for i, anc in enumerate(anchors):
            x1, y1, x2, y2 = anc
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           linewidth=1.5, edgecolor='green', facecolor='none', linestyle='-'))
            ax.text(x1, y1 - 3, f"A{i}", color='green', fontsize=6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    from time import time

    checkboard_sampling = NeighborhoodSampling(
        max_groups=4,
        max_permutations=2,
        alpha=0.0,
        cut_off_thr=0.5,
        use_anchor=True)

    image_sizes = [[1000, 800]] * 1
    boxes = [[300, 350, 350, 450]] * 1

    anchors = np.array([
        [290, 340, 310, 360],  # top-left
        [320, 330, 340, 350],  # top
        [360, 340, 380, 360],  # top-right
        [290, 370, 310, 390],  # left
        [360, 370, 380, 390],  # right
        [290, 460, 310, 480],  # bottom-left
        [320, 460, 340, 480],  # bottom
        [360, 460, 380, 480],  # bottom-right
    ], dtype=np.float32)

    checkboard_sampling.put_anchors(anchors)

    for idx, (image_size, box) in enumerate(zip(image_sizes, boxes)):
        anchor_indices = checkboard_sampling._get_anchor_box_indices(box)
        box_templates = checkboard_sampling._get_box_templates(box, anchor_indices)
        print(box_templates)

        save_path = f'debug_templates/template_{idx}.png'
        draw_box_templates(image_size, box, box_templates, anchors=anchors, save_path=save_path)
        print(f'✅ 저장 완료: {save_path}')