import numpy as np
import torch
import torch.nn.functional as F
import sys
from tqdm import tqdm
import ovdet.models.vlms.clip as CLIP
import argparse
import json

def article(name):
  return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res

single_template = [
    'a photo of {article} {}.'
]

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',
    'itap of {article} {}.',
    'itap of my {}.',
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',
    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',
    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',
    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',
    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',
    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',
    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',
    'a plastic {}.',
    'the plastic {}.',
    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',
    'an embroidered {}.',
    'the embroidered {}.',
    'a painting of the {}.',
    'a painting of a {}.',
]

BG_PROMPT_GROUPS_DEFAULT5 = {
    "sky": [
        "an empty clear blue sky background, no objects",
        "a background of sky with clouds, no objects",
        "a clean sky background, minimal, no objects",
        "featureless sky texture background, no objects",
        "open sky backdrop, no objects, no people",
        "clear sky gradient background, no objects"
    ],
    "water_surface": [
        "calm water surface background, no boats, no objects",
        "open sea water background, no objects",
        "lake water texture background, no objects",
        "ripples on water surface background, no objects",
        "smooth water background, no objects",
        "blue water surface background, no objects"
    ],
    "vegetation": [
        "dense green foliage background, no animals, no objects",
        "vegetation canopy background, no objects",
        "leafy foliage texture background, no objects",
        "forest foliage background, no objects",
        "bush leaves background, no objects",
        "green vegetation background, no objects"
    ],
    "paved_ground": [
        "paved asphalt road background, no vehicles, no objects",
        "concrete ground background, no objects",
        "asphalt pavement texture background, no objects",
        "road surface background, no vehicles, no objects",
        "sidewalk concrete background, no objects",
        "plain tarmac background, no objects"
    ],
    "plain_wall": [
        "plain concrete wall background, no objects",
        "building facade background, no signs, no objects",
        "painted wall background, no objects",
        "brick wall background, no objects",
        "smooth wall texture background, no objects",
        "exterior wall surface background, no objects"
    ],
}

@torch.no_grad()
def build_text_embedding_coco(categories, model):
    templates = multiple_templates
    zeroshot_weights, attn12_weights = [], []
    for category in categories:
        texts = [
            template.format(processed_name(category, rm_dot=True), article=article(category))
            for template in templates
        ]
        texts = ["This is " + t if t.startswith("a") or t.startswith("the") else t for t in texts]
        texts = CLIP.tokenize(texts).cuda()
        text_embeddings = model.encode_text(texts)
        text_attnfeatures, _, _ = model.encode_text_endk(texts, stepk=12, normalize=True)

        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        text_embedding = text_embeddings.mean(dim=0)
        text_embedding /= text_embedding.norm()

        text_attnfeatures = text_attnfeatures.mean(0)
        text_attnfeatures = F.normalize(text_attnfeatures, dim=0)
        attn12_weights.append(text_attnfeatures)
        zeroshot_weights.append(text_embedding)
    return torch.stack(zeroshot_weights, 0), torch.stack(attn12_weights, 0)

@torch.no_grad()
def build_text_embedding_lvis(categories, model):
    templates = multiple_templates
    zeroshot_weights, attn12_weights = [], []
    for category in tqdm(categories):
        texts = [
            template.format(processed_name(category, rm_dot=True), article=article(category))
            for template in templates
        ]
        texts = ["This is " + t if t.startswith("a") or t.startswith("the") else t for t in texts]
        texts = CLIP.tokenize(texts).cuda()
        text_embeddings = model.encode_text(texts)
        text_attnfeatures, _, _ = model.encode_text_endk(texts, stepk=12, normalize=True)

        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        text_embedding = text_embeddings.mean(dim=0)
        text_embedding /= text_embedding.norm()

        text_attnfeatures = text_attnfeatures.mean(0)
        text_attnfeatures = F.normalize(text_attnfeatures, dim=0)
        zeroshot_weights.append(text_embedding)
        attn12_weights.append(text_attnfeatures)
    return torch.stack(zeroshot_weights, 0), torch.stack(attn12_weights, 0)

@torch.no_grad()
def build_background_embeddings(model, prompt_groups):
    """
    prompt_groups: dict[str, list[str]]
      ex) {"sky": [...6~8 prompts...], "water_surface": [...], ...}
    return:
      bg_embeds: (Nb, D), bg_attn: (Nb, D), names: list of keys
    """
    names = list(prompt_groups.keys())
    bg_embeds, bg_attn = [], []
    for name in names:
        prompts = [p.strip() for p in prompt_groups[name]]
        texts = [("This is " + p) if (p.startswith("a ") or p.startswith("an ") or p.startswith("the ")) else p
                 for p in prompts]
        tokens = CLIP.tokenize(texts).cuda()

        text_embeddings = model.encode_text(tokens)                             # (M, D)
        text_attnfeatures, _, _ = model.encode_text_endk(tokens, stepk=12, normalize=True)  # (M, D)

        text_embeddings = F.normalize(text_embeddings, dim=-1)
        proto = text_embeddings.mean(dim=0)
        proto = F.normalize(proto, dim=0)

        attn_feat = text_attnfeatures.mean(0)
        attn_feat = F.normalize(attn_feat, dim=0)

        bg_embeds.append(proto)
        bg_attn.append(attn_feat)

    return torch.stack(bg_embeds, dim=0), torch.stack(bg_attn, dim=0), names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', default='checkpoints/clip_vitb16.pth') # ViT-B/16
    parser.add_argument('--ann', default='data/coco/hchoi/instances_train2017_pseudo_v0_new.json')
    parser.add_argument('--out_path', default='data/metadata/coco_clip_hand_craft_pseudo_v0_new.npy')
    parser.add_argument('--dataset', default='coco')
    parser.add_argument('--background', action='store_true',
                        help='Use 5-category background prompt ensembles to build bg embeddings')
    parser.add_argument('--bg_out_path', default='data/metadata/coco_clip_hand_craft_background.npy')
    parser.add_argument('--bg_set', default='default5', choices=['default5'],
                        help='which background prompt set to use (extend as needed)')
    args = parser.parse_args()

    model, _ = CLIP.load(name=args.model_version,
                         use_image_encoder=False,
                         download_root='checkpoints')
    model.init_weights()
    model.cuda()

    if args.background:
        if args.bg_set == 'default5':
            prompt_groups = BG_PROMPT_GROUPS_DEFAULT5
        else:
            raise ValueError(f'Unknown bg_set: {args.bg_set}')

        print(f'Building background embeddings from prompt groups: {list(prompt_groups.keys())}')
        bg_embeds, bg_attn, names = build_background_embeddings(model, prompt_groups)
        np.save(args.bg_out_path, bg_embeds.cpu().numpy())
        np.save(args.bg_out_path.replace('.npy', '_attn12.npy'), bg_attn.cpu().numpy())
        print(f'Saved bg embeddings to: {args.bg_out_path} (shape={tuple(bg_embeds.shape)})')
        print(f'Saved bg attn12 to: {args.bg_out_path.replace(".npy", "_attn12.npy")}')
        print('BG order:', names)
        sys.exit(0)

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = [x['name'] for x in sorted(data['categories'], key=lambda x: x['id'])]

    out_path = args.out_path
    if args.dataset == 'lvis':
        text_embeddings, attn12_embeddings = build_text_embedding_lvis(cat_names, model)
        np.save(out_path, text_embeddings.cpu().numpy())
        np.save(out_path.replace('.npy', '_attn12.npy'), attn12_embeddings.cpu().numpy())
    elif args.dataset == 'objects365':
        text_embeddings, attn12_embeddings = build_text_embedding_lvis(cat_names, model)
        np.save(out_path, text_embeddings.cpu().numpy())
        np.save(out_path.replace('.npy', '_attn12.npy'), attn12_embeddings.cpu().numpy())
    else:
        clip_embeddings, attn12_embeddings = build_text_embedding_coco(cat_names, model)
        np.save(out_path, clip_embeddings.cpu().numpy())
        np.save(out_path.replace('.npy', '_attn12.npy'), attn12_embeddings.cpu().numpy())
