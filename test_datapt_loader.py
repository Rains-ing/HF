import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from process_pretrain_data.hierarchical_dataset import (
    load_dataset,
    load_dataset_info,
    HierarchicalCollator,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="C:/HierarchicalBurst-FlowBERT/datasets/fine-tuning_data/hierarchical_01000001.pt")
    parser.add_argument("--vocab_path", type=str, default="C:/HierarchicalBurst-FlowBERT/datasets/fine-tuning_output_jsonl/hex_vocab.txt")
    parser.add_argument("--dataset_info", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    dataset = load_dataset(args.data)

    dataset_info = load_dataset_info(
        dataset_info_path=args.dataset_info,
        data_path=args.data,
        vocab_path=args.vocab_path,
    )

    collator = HierarchicalCollator(
        dataset_info=dataset_info,
        token_mlm_probability=0.15,
        burst_mbm_probability=0.15,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )

    print("[DATASET]")
    print("samples:", len(dataset))
    print("dataset_info:", dataset_info)

    batch = next(iter(loader))

    print("\n[BATCH SHAPES]")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
        else:
            print(k, type(v))

    print("\n[CHECK]")
    print("src[0, 0, :20]:")
    print(batch["src"][0, 0, :20])

    print("\ntoken_mbm_input[0, 0, :20]:")
    print(batch["token_mbm_input"][0, 0, :20])

    print("\ntoken_mbm_labels[0, 0, :20]:")
    print(batch["token_mbm_labels"][0, 0, :20])

    print("\ndirection[0]:")
    print(batch["direction"][0])

    print("\nburst_mask[0]:")
    print(batch["burst_mask"][0])

    print("\nburst_mbm_mask[0]:")
    print(batch["burst_mbm_mask"][0])

    print("\nmasked token count:", int(batch["token_mbm_mask"].sum().item()))
    print("masked burst count:", int(batch["burst_mbm_mask"].sum().item()))


if __name__ == "__main__":
    main()