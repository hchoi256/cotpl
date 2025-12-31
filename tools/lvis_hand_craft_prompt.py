import numpy as np

# 파일 경로
base_path = 'data/metadata/lvis_v1_clip_detpro.npy'
pseudo_path = 'data/metadata/lvis_clip_hand_craft_pseudo_v0_attn12.npy'
save_path = 'data/metadata/lvis_v1_clip_detpro_pseudo_v0_attn12.npy'

# 데이터 로드
base_embeddings = np.load(base_path)  # (1203, 512), float32
pseudo_embeddings = np.load(pseudo_path)  # (1423, 512), float16

# pseudo에서 1203 ~ 1423 인덱스 선택 (총 220개)
pseudo_new = pseudo_embeddings[1203:]  # (220, 512)

# 타입 맞추기 (float32로 변환)
pseudo_new = pseudo_new.astype(np.float32)

# base + pseudo concat
new_embeddings = np.concatenate([base_embeddings, pseudo_new], axis=0)  # (1423, 512)

# 저장
np.save(save_path, new_embeddings)

print(f"Saved new embeddings to {save_path}")
print(f"New shape: {new_embeddings.shape}, dtype: {new_embeddings.dtype}")