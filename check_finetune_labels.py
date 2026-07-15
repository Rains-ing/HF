import argparse
from collections import Counter
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", type=str, default="C:/HierarchicalBurst-FlowBERT/datasets/fine-tuning_data/hierarchical_01000001.pt")
    args = parser.parse_args()

    data = torch.load(args.pt_path, map_location="cpu")

    print("[DATASET]")
    print("samples:", len(data))

    counter = Counter()

    for sample in data:
        label = sample["label"]
        if torch.is_tensor(label):
            label = int(label.item())
        else:
            label = int(label)

        counter[label] += 1

    print("\n[LABEL DISTRIBUTION]")
    for label, count in sorted(counter.items()):
        print(f"label={label}: {count}")

    print("\n[FIRST SAMPLE]")
    first = data[0]
    for k, v in first.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
        else:
            print(k, type(v), v)


if __name__ == "__main__":
    main()