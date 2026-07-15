import os
import json
import argparse
from collections import Counter


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def iter_jsonl_files(input_path):
    """
    支持输入单个 jsonl 文件，也支持输入目录。
    如果是目录，则递归读取其中所有 .jsonl 文件。
    """
    if os.path.isfile(input_path):
        yield input_path
        return

    jsonl_files = []

    for root, dirs, files in os.walk(input_path):
        dirs.sort()
        files.sort()

        for filename in files:
            if filename.lower().endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, filename))

    jsonl_files.sort()

    for path in jsonl_files:
        yield path


def build_vocab(
    input_path,
    output_vocab,
    min_freq=1,
    max_vocab_size=None,
    lowercase=True,
):
    counter = Counter()

    total_lines = 0
    valid_lines = 0
    total_tokens = 0

    jsonl_files = list(iter_jsonl_files(input_path))

    if len(jsonl_files) == 0:
        raise FileNotFoundError(f"没有找到 jsonl 文件: {input_path}")

    print(f"[INFO] jsonl files: {len(jsonl_files)}")

    for jsonl_path in jsonl_files:
        print(f"[READ] {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as reader:
            for line in reader:
                total_lines += 1
                line = line.strip()

                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except Exception:
                    continue

                bursts = sample.get("bursts", [])

                if not isinstance(bursts, list):
                    continue

                valid_lines += 1

                for burst in bursts:
                    tokens = burst.get("tokens", [])

                    for token in tokens:
                        if lowercase:
                            token = token.lower()

                        counter[token] += 1
                        total_tokens += 1

    items = [
        (tok, freq)
        for tok, freq in counter.items()
        if freq >= min_freq and tok not in SPECIAL_TOKENS
    ]

    # 先按频率降序，再按 token 字典序，保证结果稳定
    items.sort(key=lambda x: (-x[1], x[0]))

    if max_vocab_size is not None and max_vocab_size > 0:
        keep_size = max_vocab_size - len(SPECIAL_TOKENS)
        items = items[:keep_size]

    os.makedirs(os.path.dirname(os.path.abspath(output_vocab)), exist_ok=True)

    with open(output_vocab, "w", encoding="utf-8") as writer:
        for token in SPECIAL_TOKENS:
            writer.write(token + "\n")

        for token, freq in items:
            writer.write(token + "\n")

    print("\n[DONE]")
    print(f"total json lines: {total_lines}")
    print(f"valid json lines: {valid_lines}")
    print(f"total tokens: {total_tokens}")
    print(f"unique tokens: {len(counter)}")
    print(f"kept tokens: {len(items) + len(SPECIAL_TOKENS)}")
    print(f"min_freq: {min_freq}")
    print(f"max_vocab_size: {max_vocab_size}")
    print(f"output vocab: {output_vocab}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/pre_train_output_jsonl/jsonl_parts",
        help="输入 flow_burst.jsonl 文件，或者 jsonl_parts 目录",
    )

    parser.add_argument(
        "--output_vocab",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/pre_train_output_jsonl/hex_vocab.txt",
        help="输出词表路径，例如 hex_vocab.txt",
    )

    parser.add_argument(
        "--min_freq",
        type=int,
        default=1,
        help="低于该频率的 token 不进入词表",
    )

    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=None,
        help="最大词表大小，包含特殊符号；不设置表示不限制",
    )

    parser.add_argument(
        "--no_lowercase",
        action="store_true",
        help="不把 token 转成小写。默认会转小写",
    )

    args = parser.parse_args()

    build_vocab(
        input_path=args.input,
        output_vocab=args.output_vocab,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        lowercase=not args.no_lowercase,
    )


if __name__ == "__main__":
    main()