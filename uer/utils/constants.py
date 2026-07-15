

import os
import json

DEFAULT_SPECIAL_TOKENS_MAP = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "pad_id": 0,
    "unk_id": 1,
    "cls_id": 2,
    "sep_id": 3,
    "mask_id": 4
}

special_tokens_map_path = os.path.join(
    os.getcwd(),
    "models",
    "special_tokens_map.json"
)

if os.path.exists(special_tokens_map_path):
    with open(special_tokens_map_path, mode="r", encoding="utf-8") as f:
        special_tokens_map = json.load(f)
else:
    special_tokens_map = DEFAULT_SPECIAL_TOKENS_MAP


UNK_TOKEN = special_tokens_map.get("unk_token")
CLS_TOKEN = special_tokens_map.get("cls_token")
SEP_TOKEN = special_tokens_map.get("sep_token")
MASK_TOKEN = special_tokens_map.get("mask_token")
PAD_TOKEN = special_tokens_map.get("pad_token")

# e.g. <extra_id_0>, <extra_id_1>, ... , should have consecutive IDs.
SENTINEL_TOKEN = special_tokens_map.get("sentinel_token")
