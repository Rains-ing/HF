import os
import json
import argparse
import random
from collections import Counter, defaultdict

import torch


def find_pt_files(input_dir):
    pt_files = []

    for root, dirs, files in os.walk(input_dir):
        for name in files:
            if name.endswith(".pt"):
                pt_files.append(os.path.join(root, name))

    pt_files.sort()
    return pt_files


def get_label(sample):
    label = sample.get("label", None)

    if label is None:
        raise ValueError("样本中缺少 label 字段")

    if torch.is_tensor(label):
        return int(label.item())

    return int(label)


def load_all_samples(pt_files):
    all_samples = []
    file_stats = []

    for idx, pt_path in enumerate(pt_files, start=1):
        print(f"[LOAD {idx}/{len(pt_files)}] {pt_path}")

        data = torch.load(pt_path, map_location="cpu")

        if not isinstance(data, list):
            raise ValueError(f"{pt_path} 不是 list 类型 .pt 数据集")

        counter = Counter()

        for sample in data:
            label = get_label(sample)
            counter[label] += 1
            all_samples.append(sample)

        file_stats.append({
            "path": pt_path,
            "num_samples": len(data),
            "label_counter": dict(counter),
        })

        print(f"  samples={len(data)} labels={dict(counter)}")

        del data

    return all_samples, file_stats


def stratified_split(samples, train_ratio, valid_ratio, test_ratio, seed):
    """
    按 label 分层划分。
    保证每个类别都按比例进入 train / valid / test。
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)

    label_to_samples = defaultdict(list)

    for sample in samples:
        label = get_label(sample)
        label_to_samples[label].append(sample)

    train_set = []
    valid_set = []
    test_set = []

    split_stats = {}

    for label, items in sorted(label_to_samples.items()):
        rng.shuffle(items)

        n = len(items)

        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        train_items = items[:n_train]
        valid_items = items[n_train:n_train + n_valid]
        test_items = items[n_train + n_valid:]

        train_set.extend(train_items)
        valid_set.extend(valid_items)
        test_set.extend(test_items)

        split_stats[label] = {
            "total": n,
            "train": len(train_items),
            "valid": len(valid_items),
            "test": len(test_items),
        }

    # 各集合内部再整体打乱
    rng.shuffle(train_set)
    rng.shuffle(valid_set)
    rng.shuffle(test_set)

    return train_set, valid_set, test_set, split_stats


def count_labels(samples):
    counter = Counter()

    for sample in samples:
        counter[get_label(sample)] += 1

    return dict(sorted(counter.items()))


def save_pt(samples, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(samples, path)
    print(f"[SAVE] {path} samples={len(samples)} labels={count_labels(samples)}")


def inspect_first_sample(samples):
    if len(samples) == 0:
        return {}

    first = samples[0]
    info = {}

    for k, v in first.items():
        if torch.is_tensor(v):
            info[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
            }
        else:
            info[k] = {
                "type": str(type(v)),
                "value": v,
            }

    return info


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/fine-tuning_data_0604_cstnet-tls_only_de_ip",
        help="包含多个 .pt 文件的文件夹",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/finetuning-data_reset_0604_cstnet-tls_only_de_ip",
        help="输出 train.pt / valid.pt / test.pt 的目录",
    )

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    pt_files = find_pt_files(args.input_dir)

    if len(pt_files) == 0:
        raise ValueError(f"没有在目录中找到 .pt 文件: {args.input_dir}")

    print("[PT FILES]")
    for p in pt_files:
        print(p)

    all_samples, file_stats = load_all_samples(pt_files)

    print("\n[ALL DATA]")
    print("total samples:", len(all_samples))
    print("label distribution:", count_labels(all_samples))

    train_set, valid_set, test_set, split_stats = stratified_split(
        samples=all_samples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_path = os.path.join(args.output_dir, "train.pt")
    valid_path = os.path.join(args.output_dir, "valid.pt")
    test_path = os.path.join(args.output_dir, "test.pt")

    save_pt(train_set, train_path)
    save_pt(valid_set, valid_path)
    save_pt(test_set, test_path)

    info = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "num_input_pt_files": len(pt_files),
        "total_samples": len(all_samples),
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "all_label_counter": count_labels(all_samples),
        "train_label_counter": count_labels(train_set),
        "valid_label_counter": count_labels(valid_set),
        "test_label_counter": count_labels(test_set),
        "split_stats_by_label": split_stats,
        "file_stats": file_stats,
        "first_sample_info": inspect_first_sample(all_samples),
    }

    info_path = os.path.join(args.output_dir, "split_info.json")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[SAVE INFO] {info_path}")

    print("\n[DONE]")
    print("train:", len(train_set), count_labels(train_set))
    print("valid:", len(valid_set), count_labels(valid_set))
    print("test :", len(test_set), count_labels(test_set))


if __name__ == "__main__":
    main()