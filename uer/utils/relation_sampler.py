import torch


def get_valid_burst_indices(burst_mask):
    """
    burst_mask: [B, N]
    返回每个 batch 样本中真实 burst 的下标列表。
    """
    B, N = burst_mask.shape
    valid = []

    for b in range(B):
        idx = torch.nonzero(burst_mask[b].bool(), as_tuple=False).view(-1)
        valid.append(idx)

    return valid


def sample_nsp_pairs(
    burst_mask,
    num_pairs=64,
    positive_ratio=0.5,
):
    """
    构造 NSP 二分类样本。

    label:
        1 = same flow 且相邻顺序正确
        0 = negative pair

    返回:
        pairs:  [M, 4] = b1, n1, b2, n2
        labels: [M]
    """
    device = burst_mask.device
    B, N = burst_mask.shape
    valid = get_valid_burst_indices(burst_mask)

    pairs = []
    labels = []

    max_try = num_pairs * 20
    tries = 0

    while len(pairs) < num_pairs and tries < max_try:
        tries += 1

        make_positive = torch.rand(1).item() < positive_ratio

        if make_positive:
            # 正样本：同一个 flow 中相邻 burst
            b = torch.randint(0, B, (1,), device=device).item()
            idx = valid[b]

            if len(idx) < 2:
                continue

            # 找相邻位置：使用真实 burst 序列中的相邻
            pos = torch.randint(0, len(idx) - 1, (1,), device=device).item()
            n1 = idx[pos].item()
            n2 = idx[pos + 1].item()

            pairs.append([b, n1, b, n2])
            labels.append(1)

        else:
            # 负样本：优先不同 flow
            if B < 2:
                continue

            b1 = torch.randint(0, B, (1,), device=device).item()
            b2 = torch.randint(0, B, (1,), device=device).item()

            if b1 == b2:
                continue

            idx1 = valid[b1]
            idx2 = valid[b2]

            if len(idx1) == 0 or len(idx2) == 0:
                continue

            n1 = idx1[torch.randint(0, len(idx1), (1,), device=device)].item()
            n2 = idx2[torch.randint(0, len(idx2), (1,), device=device)].item()

            pairs.append([b1, n1, b2, n2])
            labels.append(0)

    if len(pairs) == 0:
        return None, None

    return (
        torch.tensor(pairs, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def sample_sodf_pairs(
    burst_mask,
    direction,
    num_pairs=64,
):
    """
    构造 SODF 五分类样本。

    label:
        0：同一 flow，同方向，顺序正确
        1：同一 flow，同方向，顺序错误
        2：同一 flow，不同方向
        3：不同 flow，同方向
        4：不同 flow，不同方向

    返回:
        pairs:  [M, 4] = b1, n1, b2, n2
        labels: [M]
    """
    device = burst_mask.device
    B, N = burst_mask.shape
    valid = get_valid_burst_indices(burst_mask)

    pairs = []
    labels = []

    max_try = num_pairs * 50
    tries = 0

    while len(pairs) < num_pairs and tries < max_try:
        tries += 1

        cls = torch.randint(0, 5, (1,), device=device).item()

        if cls in [0, 1, 2]:
            # 同一 flow
            b = torch.randint(0, B, (1,), device=device).item()
            idx = valid[b]

            if len(idx) < 2:
                continue

            i = torch.randint(0, len(idx), (1,), device=device).item()
            j = torch.randint(0, len(idx), (1,), device=device).item()

            if i == j:
                continue

            n1 = idx[i].item()
            n2 = idx[j].item()

            d1 = int(direction[b, n1].item())
            d2 = int(direction[b, n2].item())

            if d1 not in [0, 1] or d2 not in [0, 1]:
                continue

            if cls == 0:
                # 同 flow，同方向，顺序正确
                if d1 != d2:
                    continue
                if n1 >= n2:
                    continue

                pairs.append([b, n1, b, n2])
                labels.append(0)

            elif cls == 1:
                # 同 flow，同方向，顺序错误
                if d1 != d2:
                    continue
                if n1 <= n2:
                    continue

                pairs.append([b, n1, b, n2])
                labels.append(1)

            else:
                # cls == 2，同 flow，不同方向
                if d1 == d2:
                    continue

                pairs.append([b, n1, b, n2])
                labels.append(2)

        else:
            # 不同 flow
            if B < 2:
                continue

            b1 = torch.randint(0, B, (1,), device=device).item()
            b2 = torch.randint(0, B, (1,), device=device).item()

            if b1 == b2:
                continue

            idx1 = valid[b1]
            idx2 = valid[b2]

            if len(idx1) == 0 or len(idx2) == 0:
                continue

            n1 = idx1[torch.randint(0, len(idx1), (1,), device=device)].item()
            n2 = idx2[torch.randint(0, len(idx2), (1,), device=device)].item()

            d1 = int(direction[b1, n1].item())
            d2 = int(direction[b2, n2].item())

            if d1 not in [0, 1] or d2 not in [0, 1]:
                continue

            if cls == 3:
                # 不同 flow，同方向
                if d1 != d2:
                    continue

                pairs.append([b1, n1, b2, n2])
                labels.append(3)

            else:
                # cls == 4，不同 flow，不同方向
                if d1 == d2:
                    continue

                pairs.append([b1, n1, b2, n2])
                labels.append(4)

    if len(pairs) == 0:
        return None, None

    return (
        torch.tensor(pairs, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )