import torch
from torch.utils.data import Dataset


class FinetunePTDataset(Dataset):
    """
    读取微调阶段的 .pt 文件。

    每条样本需要包含：
        src:        [N, L]
        direction:  [N]
        burst_mask: [N]
        token_mask: [N, L]
        label:      scalar
    """

    def __init__(self, pt_path):
        super().__init__()
        self.pt_path = pt_path
        self.samples = torch.load(pt_path, map_location="cpu")

        if not isinstance(self.samples, list):
            raise ValueError(f"{pt_path} 不是 list 类型 .pt 数据集")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class FinetuneCollator:
    """
    微调阶段 collate_fn。

    注意：
    微调不做 token-MBM
    微调不做 burst-MBM
    只堆叠 clean 输入和 label
    """

    def __call__(self, samples):
        batch = {
            "src": torch.stack([x["src"] for x in samples], dim=0),
            "direction": torch.stack([x["direction"] for x in samples], dim=0),
            "burst_mask": torch.stack([x["burst_mask"] for x in samples], dim=0),
            "token_mask": torch.stack([x["token_mask"] for x in samples], dim=0),
            "label": torch.stack([x["label"] for x in samples], dim=0).long(),
        }

        return batch