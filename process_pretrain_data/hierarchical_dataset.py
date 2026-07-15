import os
import json
import bisect

import torch
from torch.utils.data import Dataset


class SinglePTDataset(Dataset):
    """
    读取单个 .pt 数据集。
    适合 debug 或小规模数据。
    """

    def __init__(self, pt_path):
        super().__init__()
        self.pt_path = pt_path
        self.samples = torch.load(pt_path, map_location="cpu")

        if not isinstance(self.samples, list):
            raise ValueError(f"{pt_path} 不是 list 类型数据集")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ShardedPTDataset(Dataset):
    """
    读取多个 .pt 分片。

    输入是 pt_manifest.jsonl，例如：
    {"shard_index":1,"path":"xxx.pt","num_samples":50000}
    """

    def __init__(self, manifest_path):
        super().__init__()
        self.manifest_path = manifest_path
        self.shards = self._load_manifest(manifest_path)

        if len(self.shards) == 0:
            raise ValueError(f"manifest 中没有 shard: {manifest_path}")

        self.cumulative_sizes = []
        total = 0

        for shard in self.shards:
            total += shard["num_samples"]
            self.cumulative_sizes.append(total)

        self.total_samples = total

        self._cached_shard_idx = None
        self._cached_samples = None

    def _load_manifest(self, manifest_path):
        shards = []
        base_dir = os.path.dirname(os.path.abspath(manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                path = item["path"]

                if not os.path.isabs(path):
                    path = os.path.join(base_dir, path)

                shards.append({
                    "shard_index": int(item["shard_index"]),
                    "path": path,
                    "num_samples": int(item["num_samples"]),
                    "jsonl_index": int(item.get("jsonl_index", item["shard_index"])),
                    "local_shard_id": int(item.get("local_shard_id", 1)),
                })

        # 保证读取顺序稳定
        shards.sort(key=lambda x: (x["jsonl_index"], x["local_shard_id"], x["shard_index"]))
        return shards

    def __len__(self):
        return self.total_samples

    def _find_shard(self, global_idx):
        shard_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)

        if shard_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_sizes[shard_idx - 1]

        return shard_idx, local_idx

    def _load_shard(self, shard_idx):
        if self._cached_shard_idx == shard_idx and self._cached_samples is not None:
            return self._cached_samples

        shard_path = self.shards[shard_idx]["path"]
        samples = torch.load(shard_path, map_location="cpu")

        if not isinstance(samples, list):
            raise ValueError(f"{shard_path} 不是 list 类型 shard")

        self._cached_shard_idx = shard_idx
        self._cached_samples = samples

        return samples

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.total_samples + idx

        if idx < 0 or idx >= self.total_samples:
            raise IndexError(idx)

        shard_idx, local_idx = self._find_shard(idx)
        samples = self._load_shard(shard_idx)

        return samples[local_idx]


def load_dataset(data_path):
    """
    自动判断读取单个 .pt 还是 pt_manifest.jsonl。
    """
    if data_path.endswith(".pt"):
        return SinglePTDataset(data_path)

    if data_path.endswith(".jsonl"):
        return ShardedPTDataset(data_path)

    raise ValueError(
        "data_path 必须是 .pt 文件或 pt_manifest.jsonl 文件，"
        f"当前为: {data_path}"
    )


def load_vocab(vocab_path):
    """
    读取 vocab.txt，一行一个 token。
    要求包含：
        [PAD]
        [UNK]
        [CLS]
        [SEP]
        [MASK]
    """
    token_to_id = {}

    with open(vocab_path, "r", encoding="utf-8") as reader:
        for line in reader:
            token = line.strip()
            if not token:
                continue
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

    required = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    for token in required:
        if token not in token_to_id:
            raise ValueError(f"vocab 中缺少特殊符号: {token}")

    special_ids = {
        "pad_id": token_to_id["[PAD]"],
        "unk_id": token_to_id["[UNK]"],
        "cls_id": token_to_id["[CLS]"],
        "sep_id": token_to_id["[SEP]"],
        "mask_id": token_to_id["[MASK]"],
    }

    return token_to_id, special_ids


def infer_dataset_info_from_pt(data_path, vocab_path):
    """
    如果没有 dataset_info.json，就从 .pt 和 vocab.txt 推断：
        vocab_size
        special_ids
        max_burst_num
        max_burst_len
    """
    dataset = load_dataset(data_path)
    first = dataset[0]

    if "src" not in first:
        raise ValueError("pt 样本中缺少 src 字段")

    max_burst_num, max_burst_len = first["src"].shape

    vocab, special_ids = load_vocab(vocab_path)

    info = {
        "vocab_size": len(vocab),
        "special_ids": special_ids,
        "max_burst_num": int(max_burst_num),
        "max_burst_len": int(max_burst_len),
    }

    return info


def load_dataset_info(dataset_info_path=None, data_path=None, vocab_path=None):
    """
    优先读取 dataset_info.json。
    如果没有 dataset_info.json，则通过 .pt 和 vocab.txt 自动推断。
    """
    if dataset_info_path is not None and dataset_info_path != "" and os.path.exists(dataset_info_path):
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        required = ["vocab_size", "special_ids", "max_burst_num", "max_burst_len"]

        for key in required:
            if key not in info:
                raise ValueError(f"dataset_info.json 缺少字段: {key}")

        return info

    if data_path is None or vocab_path is None:
        raise ValueError("没有 dataset_info.json 时，必须提供 data_path 和 vocab_path")

    return infer_dataset_info_from_pt(data_path, vocab_path)

def create_burst_mbm_mask(
    burst_mask,
    mbm_probability=0.15,
):
    """
    生成 flow-level burst MBM mask。

    burst_mask: [B, N]
        1 = 真实 burst
        0 = padding burst

    返回:
        burst_mbm_mask: [B, N]
    """
    device = burst_mask.device
    valid_bursts = burst_mask.bool()

    prob = torch.full(
        burst_mask.shape,
        mbm_probability,
        dtype=torch.float,
        device=device,
    )

    prob = prob * valid_bursts.float()

    burst_mbm_mask = torch.bernoulli(prob).bool()

    # 每个 flow 至少 mask 一个真实 burst
    B, N = burst_mask.shape

    for b in range(B):
        valid_indices = torch.nonzero(valid_bursts[b], as_tuple=False).view(-1)

        if len(valid_indices) == 0:
            continue

        if burst_mbm_mask[b].sum() == 0:
            chosen = valid_indices[
                torch.randint(0, len(valid_indices), (1,), device=device)
            ]
            burst_mbm_mask[b, chosen] = True

        # 如果一个 flow 有多个真实 burst，不要全部 mask 掉。
        # 这样 token-level MBM 仍然有机会在未被 burst-MBM 覆盖的位置产生监督。
        if len(valid_indices) > 1 and burst_mbm_mask[b, valid_indices].all():
            keep = valid_indices[
                torch.randint(0, len(valid_indices), (1,), device=device)
            ]
            burst_mbm_mask[b, keep] = False

    return burst_mbm_mask


def create_token_mbm_inputs(
    src,
    token_mask,
    special_ids,
    vocab_size,
    mlm_probability=0.15,
    forbid_burst_mask=None,
):
    """
    生成 token-level MBM 输入。

    src: [B, N, L]
    token_mask: [B, N, L]

    forbid_burst_mask:
        [B, N]
        如果某个 burst 已经被 burst-level MBM 选中，
        可以禁止它内部再做 token-level MBM。

    返回:
        token_mbm_input:  [B, N, L]
        token_mbm_labels: [B, N, L]
        token_mbm_mask:   [B, N, L]
    """
    pad_id = int(special_ids["pad_id"])
    cls_id = int(special_ids["cls_id"])
    sep_id = int(special_ids["sep_id"])
    mask_id = int(special_ids["mask_id"])

    token_mbm_input = src.clone()
    token_mbm_labels = torch.full_like(src, fill_value=-100)

    base_valid_positions = token_mask.bool()
    base_valid_positions = base_valid_positions & (src != pad_id)
    base_valid_positions = base_valid_positions & (src != cls_id)
    base_valid_positions = base_valid_positions & (src != sep_id)
    base_valid_positions = base_valid_positions & (src != mask_id)

    valid_positions = base_valid_positions

    # 避免 token-MBM 和 burst-MBM 重叠
    if forbid_burst_mask is not None:
        forbid_positions = forbid_burst_mask.unsqueeze(-1).expand_as(src).bool()
        valid_positions = valid_positions & (~forbid_positions)

        # 极短 flow 可能所有普通 token 都落在 burst-MBM 中。
        # 这种情况下退回到允许重叠，优先保证 token-MBM 有监督信号。
        if not valid_positions.any():
            valid_positions = base_valid_positions

    probability_matrix = torch.full(
        src.shape,
        mlm_probability,
        dtype=torch.float,
        device=src.device,
    )
    probability_matrix = probability_matrix * valid_positions.float()

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # batch 内至少 mask 一个普通 token，避免全 -100 label 导致 token loss 为 0/NaN。
    if not masked_indices.any() and valid_positions.any():
        valid_flat = torch.nonzero(valid_positions.reshape(-1), as_tuple=False).view(-1)
        chosen = valid_flat[
            torch.randint(0, len(valid_flat), (1,), device=src.device)
        ]
        masked_indices.reshape(-1)[chosen] = True

    token_mbm_labels[masked_indices] = src[masked_indices]

    # 80% 替换为 [MASK]
    replace_with_mask = (
        torch.bernoulli(torch.full(src.shape, 0.8, dtype=torch.float, device=src.device)).bool()
        & masked_indices
    )
    token_mbm_input[replace_with_mask] = mask_id

    # 10% 替换为随机 token
    replace_with_random = (
        torch.bernoulli(torch.full(src.shape, 0.5, dtype=torch.float, device=src.device)).bool()
        & masked_indices
        & ~replace_with_mask
    )

    random_words = torch.randint(
        low=0,
        high=vocab_size,
        size=src.shape,
        dtype=torch.long,
        device=src.device,
    )

    token_mbm_input[replace_with_random] = random_words[replace_with_random]

    return token_mbm_input, token_mbm_labels, masked_indices


class HierarchicalCollator:
    """
    DataLoader 的 collate_fn。

    作用：
    1. 把单条样本堆叠成 batch
    2. 动态生成 burst-level MBM mask
    3. 动态生成 token-level MBM input/labels

    注意：
    默认先生成 burst_mbm_mask；
    被 burst_mbm_mask 选中的 burst，不再做 token-level mask。
    """

    def __init__(
        self,
        dataset_info,
        token_mlm_probability=0.15,
        burst_mbm_probability=0.15,
        avoid_token_burst_overlap=True,
    ):
        self.dataset_info = dataset_info
        self.special_ids = dataset_info["special_ids"]
        self.vocab_size = int(dataset_info["vocab_size"])
        self.token_mlm_probability = token_mlm_probability
        self.burst_mbm_probability = burst_mbm_probability
        self.avoid_token_burst_overlap = avoid_token_burst_overlap

    def __call__(self, samples):
        src = torch.stack([x["src"] for x in samples], dim=0)
        direction = torch.stack([x["direction"] for x in samples], dim=0)
        burst_mask = torch.stack([x["burst_mask"] for x in samples], dim=0)
        token_mask = torch.stack([x["token_mask"] for x in samples], dim=0)
        label = torch.stack([x["label"] for x in samples], dim=0)

        # 先生成 burst-level MBM mask
        burst_mbm_mask = create_burst_mbm_mask(
            burst_mask=burst_mask,
            mbm_probability=self.burst_mbm_probability,
        )

        forbid_burst_mask = burst_mbm_mask if self.avoid_token_burst_overlap else None

        # 再生成 token-level MBM，避免和 burst-level MBM 重叠
        token_mbm_input, token_mbm_labels, token_mbm_mask = create_token_mbm_inputs(
            src=src,
            token_mask=token_mask,
            special_ids=self.special_ids,
            vocab_size=self.vocab_size,
            mlm_probability=self.token_mlm_probability,
            forbid_burst_mask=forbid_burst_mask,
        )

        batch = {
            # clean data
            "src": src,
            "direction": direction,
            "burst_mask": burst_mask,
            "token_mask": token_mask,
            "label": label,

            # token-level MBM
            "token_mbm_input": token_mbm_input,
            "token_mbm_labels": token_mbm_labels,
            "token_mbm_mask": token_mbm_mask,

            # burst-level MBM
            "burst_mbm_mask": burst_mbm_mask,
        }

        return batch
