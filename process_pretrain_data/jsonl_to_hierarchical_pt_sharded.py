import os
import gc
import json
import argparse
import multiprocessing as mp
from types import SimpleNamespace
from collections import Counter

import torch

try:
    from tokenizers import Tokenizer as HFTokenizer
except ImportError:
    HFTokenizer = None


SPECIAL_TOKENS = {
    "pad": "[PAD]",
    "unk": "[UNK]",
    "cls": "[CLS]",
    "sep": "[SEP]",
    "mask": "[MASK]",
}


_GLOBAL_VOCAB = None
_GLOBAL_SPECIAL_IDS = None
_GLOBAL_LABEL_MAP = None
_GLOBAL_TOKENIZER_TYPE = "hex"
_GLOBAL_BPE_TOKENIZER = None
_GLOBAL_BPE_TOKENIZER_MODEL_TYPE = None


def iter_jsonl_files(input_path):
    if os.path.isfile(input_path):
        yield os.path.abspath(input_path)
        return

    jsonl_files = []

    for root, dirs, files in os.walk(input_path):
        dirs.sort()
        files.sort()

        for filename in files:
            if filename.lower().endswith(".jsonl"):
                jsonl_files.append(os.path.abspath(os.path.join(root, filename)))

    jsonl_files.sort()

    for path in jsonl_files:
        yield path


def load_vocab(vocab_path, lowercase=True, verbose=True):
    token_to_id = {}

    with open(vocab_path, "r", encoding="utf-8") as reader:
        for line in reader:
            token = line.strip()

            if not token:
                continue

            if lowercase and token not in SPECIAL_TOKENS.values():
                token = token.lower()

            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

    for name, tok in SPECIAL_TOKENS.items():
        if tok not in token_to_id:
            raise ValueError(f"vocab 中缺少特殊符号: {tok}")

    special_ids = {
        "pad_id": token_to_id[SPECIAL_TOKENS["pad"]],
        "unk_id": token_to_id[SPECIAL_TOKENS["unk"]],
        "cls_id": token_to_id[SPECIAL_TOKENS["cls"]],
        "sep_id": token_to_id[SPECIAL_TOKENS["sep"]],
        "mask_id": token_to_id[SPECIAL_TOKENS["mask"]],
    }

    if verbose:
        print("[VOCAB]")
        print(f"vocab size: {len(token_to_id)}")
        print(f"special ids: {special_ids}")

    return token_to_id, special_ids


def build_or_load_label_map(input_path, label_map_path=None, lowercase_label=False):
    if label_map_path is not None and os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        print(f"[LABEL] loaded label_map: {label_map_path}")
        return label_map

    labels = set()

    for jsonl_path in iter_jsonl_files(input_path):
        with open(jsonl_path, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except Exception:
                    continue

                label = sample.get("label", "unknown")
                if label is None:
                    label = "unknown"

                label = str(label).strip()
                if lowercase_label:
                    label = label.lower()

                if label == "" or label.lower() == "unknown":
                    continue

                labels.add(label)

    labels = sorted(labels)
    label_map = {label: idx for idx, label in enumerate(labels)}

    print("[LABEL] built label_map:")
    print(label_map)

    return label_map


def token_to_id(token, vocab, unk_id, lowercase=True):
    if lowercase:
        token = token.lower()
    return vocab.get(token, unk_id)


def tokens_to_hex_string(tokens, lowercase=True):
    if lowercase:
        tokens = [str(t).lower() for t in tokens]
    else:
        tokens = [str(t) for t in tokens]

    return "".join(tokens)


def tokens_to_etbert_bigram_tokens(tokens, lowercase=True):
    hex_string = tokens_to_hex_string(tokens, lowercase=lowercase)

    if len(hex_string) % 2 == 1:
        hex_string = hex_string[:-1]

    byte_tokens = [
        hex_string[i:i + 2]
        for i in range(0, len(hex_string), 2)
        if len(hex_string[i:i + 2]) == 2
    ]

    return [
        byte_tokens[i] + byte_tokens[i + 1]
        for i in range(len(byte_tokens) - 1)
    ]


def tokens_to_etbert_bigram_text(tokens, lowercase=True):
    return " ".join(tokens_to_etbert_bigram_tokens(tokens, lowercase=lowercase))


def infer_bpe_tokenizer_model_type(bpe_tokenizer_path):
    if not bpe_tokenizer_path:
        return None

    try:
        with open(bpe_tokenizer_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model", {}).get("type", "").lower() or None
    except Exception:
        return None


def load_bpe_tokenizer(bpe_tokenizer_path):
    if HFTokenizer is None:
        raise ImportError("需要安装 tokenizers 才能使用 BPE tokenizer")

    if bpe_tokenizer_path is None or bpe_tokenizer_path == "":
        raise ValueError("tokenizer_type=bpe 时必须提供 bpe_tokenizer_path")

    if not os.path.exists(bpe_tokenizer_path):
        raise FileNotFoundError(bpe_tokenizer_path)

    return HFTokenizer.from_file(bpe_tokenizer_path)


def convert_burst_to_ids(
    burst_tokens,
    vocab,
    special_ids,
    max_burst_len,
    lowercase=True,
    add_cls_sep=True,
    tokenizer_type="hex",
    bpe_tokenizer=None,
    bpe_tokenizer_model_type=None,
):
    pad_id = special_ids["pad_id"]
    unk_id = special_ids["unk_id"]
    cls_id = special_ids["cls_id"]
    sep_id = special_ids["sep_id"]

    if add_cls_sep and max_burst_len < 2:
        raise ValueError("max_burst_len 至少需要 >= 2，才能放 [CLS] 和 [SEP]")

    ids = []

    if add_cls_sep:
        ids.append(cls_id)
        max_real_tokens = max_burst_len - 2
    else:
        max_real_tokens = max_burst_len

    tokenizer_type = (tokenizer_type or "hex").lower()

    if tokenizer_type == "hex":
        burst_tokens = burst_tokens[:max_real_tokens]

        for tok in burst_tokens:
            ids.append(token_to_id(tok, vocab, unk_id, lowercase=lowercase))
    elif tokenizer_type == "bpe":
        if bpe_tokenizer is None:
            raise ValueError("tokenizer_type=bpe 时 worker 未加载 BPE tokenizer")

        if (bpe_tokenizer_model_type or "").lower() == "wordpiece":
            text = tokens_to_etbert_bigram_text(burst_tokens, lowercase=lowercase)
        else:
            text = tokens_to_hex_string(burst_tokens, lowercase=lowercase)

        real_ids = bpe_tokenizer.encode(text, add_special_tokens=False).ids if text else []
        ids.extend(real_ids[:max_real_tokens])
    else:
        raise ValueError(f"未知 tokenizer_type: {tokenizer_type}")

    if add_cls_sep:
        ids.append(sep_id)

    ids = ids[:max_burst_len]
    token_mask = [1] * len(ids)

    while len(ids) < max_burst_len:
        ids.append(pad_id)
        token_mask.append(0)

    return ids, token_mask


def convert_sample(
    raw_sample,
    vocab,
    special_ids,
    label_map,
    max_burst_num,
    max_burst_len,
    lowercase=True,
    lowercase_label=False,
    add_cls_sep=True,
    keep_meta=False,
    tokenizer_type="hex",
    bpe_tokenizer=None,
    bpe_tokenizer_model_type=None,
):
    bursts = raw_sample.get("bursts", [])

    if not isinstance(bursts, list) or len(bursts) == 0:
        return None

    src = []
    token_masks = []
    directions = []
    burst_mask = []

    for burst in bursts[:max_burst_num]:
        tokens = burst.get("tokens", [])

        if not isinstance(tokens, list) or len(tokens) == 0:
            continue

        ids, tok_mask = convert_burst_to_ids(
            burst_tokens=tokens,
            vocab=vocab,
            special_ids=special_ids,
            max_burst_len=max_burst_len,
            lowercase=lowercase,
            add_cls_sep=add_cls_sep,
            tokenizer_type=tokenizer_type,
            bpe_tokenizer=bpe_tokenizer,
            bpe_tokenizer_model_type=bpe_tokenizer_model_type,
        )

        direction = burst.get("dir", 0)

        try:
            direction = int(direction)
        except Exception:
            direction = 0

        if direction not in [0, 1]:
            direction = 0

        src.append(ids)
        token_masks.append(tok_mask)
        directions.append(direction)
        burst_mask.append(1)

    if len(src) == 0:
        return None

    pad_id = special_ids["pad_id"]

    while len(src) < max_burst_num:
        src.append([pad_id] * max_burst_len)
        token_masks.append([0] * max_burst_len)
        directions.append(2)       # 2 = padding burst
        burst_mask.append(0)

    label_name = raw_sample.get("label", "unknown")
    if label_name is None:
        label_name = "unknown"

    label_name = str(label_name).strip()

    if lowercase_label:
        label_name = label_name.lower()

    if label_name == "" or label_name.lower() == "unknown":
        label_id = -1
    else:
        label_id = label_map.get(label_name, -1)

    sample = {
        "src": torch.tensor(src, dtype=torch.long),
        "direction": torch.tensor(directions, dtype=torch.long),
        "burst_mask": torch.tensor(burst_mask, dtype=torch.long),
        "token_mask": torch.tensor(token_masks, dtype=torch.long),
        "label": torch.tensor(label_id, dtype=torch.long),
    }

    # 为了节省内存和磁盘，默认不保存字符串元信息
    if keep_meta:
        sample["flow_id"] = raw_sample.get("flow_id", "")
        sample["pcap"] = raw_sample.get("pcap", "")
        sample["label_name"] = label_name

    return sample


def init_worker(
    vocab_path,
    label_map,
    lowercase,
    tokenizer_type="hex",
    bpe_tokenizer_path=None,
):
    global _GLOBAL_VOCAB
    global _GLOBAL_SPECIAL_IDS
    global _GLOBAL_LABEL_MAP
    global _GLOBAL_TOKENIZER_TYPE
    global _GLOBAL_BPE_TOKENIZER
    global _GLOBAL_BPE_TOKENIZER_MODEL_TYPE

    _GLOBAL_VOCAB, _GLOBAL_SPECIAL_IDS = load_vocab(
        vocab_path=vocab_path,
        lowercase=lowercase,
        verbose=False,
    )
    _GLOBAL_LABEL_MAP = label_map
    _GLOBAL_TOKENIZER_TYPE = (tokenizer_type or "hex").lower()
    _GLOBAL_BPE_TOKENIZER = None
    _GLOBAL_BPE_TOKENIZER_MODEL_TYPE = None

    if _GLOBAL_TOKENIZER_TYPE == "bpe":
        _GLOBAL_BPE_TOKENIZER = load_bpe_tokenizer(bpe_tokenizer_path)
        _GLOBAL_BPE_TOKENIZER_MODEL_TYPE = infer_bpe_tokenizer_model_type(bpe_tokenizer_path)
        print(f"[TOKENIZER] bpe_tokenizer_model_type={_GLOBAL_BPE_TOKENIZER_MODEL_TYPE}")
    elif _GLOBAL_TOKENIZER_TYPE != "hex":
        raise ValueError(f"未知 tokenizer_type: {_GLOBAL_TOKENIZER_TYPE}")


def save_shard(samples, output_dir, shard_global_id, prefix):
    os.makedirs(output_dir, exist_ok=True)

    shard_path = os.path.join(
        output_dir,
        f"{prefix}_{shard_global_id:08d}.pt"
    )

    torch.save(samples, shard_path)

    return shard_path


def process_jsonl_file(task):
    """
    一个 worker 处理一个 jsonl 文件。

    处理过程中：
    - 逐行读 JSONL
    - 转成 tensor sample
    - 满 shard_size 就保存一个 .pt 分片
    - 保存后清空 samples 并 gc.collect()
    """
    global _GLOBAL_VOCAB
    global _GLOBAL_SPECIAL_IDS
    global _GLOBAL_LABEL_MAP
    global _GLOBAL_BPE_TOKENIZER_MODEL_TYPE

    jsonl_index = task["jsonl_index"]
    jsonl_path = task["jsonl_path"]
    args = SimpleNamespace(**task["args_dict"])

    result = {
        "jsonl_index": jsonl_index,
        "jsonl_path": jsonl_path,
        "total_lines": 0,
        "valid_samples": 0,
        "skipped_samples": 0,
        "shards": [],
        "label_counter": {},
        "error": None,
    }

    samples = []
    label_counter = Counter()
    local_shard_id = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as reader:
            for line in reader:
                result["total_lines"] += 1

                line = line.strip()
                if not line:
                    result["skipped_samples"] += 1
                    continue

                try:
                    raw_sample = json.loads(line)
                except Exception:
                    result["skipped_samples"] += 1
                    continue

                sample = convert_sample(
                    raw_sample=raw_sample,
                    vocab=_GLOBAL_VOCAB,
                    special_ids=_GLOBAL_SPECIAL_IDS,
                    label_map=_GLOBAL_LABEL_MAP,
                    max_burst_num=args.max_burst_num,
                    max_burst_len=args.max_burst_len,
                    lowercase=not args.no_lowercase,
                    lowercase_label=args.lowercase_label,
                    add_cls_sep=not args.no_cls_sep,
                    keep_meta=args.keep_meta,
                    tokenizer_type=_GLOBAL_TOKENIZER_TYPE,
                    bpe_tokenizer=_GLOBAL_BPE_TOKENIZER,
                    bpe_tokenizer_model_type=_GLOBAL_BPE_TOKENIZER_MODEL_TYPE,
                )

                if sample is None:
                    result["skipped_samples"] += 1
                    continue

                samples.append(sample)
                result["valid_samples"] += 1
                label_counter[str(sample["label"].item())] += 1

                if len(samples) >= args.shard_size:
                    local_shard_id += 1

                    # 这里用 jsonl_index + local_shard_id 保证文件名稳定、不冲突
                    shard_name_id = jsonl_index * 1000000 + local_shard_id

                    shard_path = save_shard(
                        samples=samples,
                        output_dir=args.output_dir,
                        shard_global_id=shard_name_id,
                        prefix=args.output_prefix,
                    )

                    result["shards"].append({
                        "path": shard_path,
                        "num_samples": len(samples),
                        "jsonl_index": jsonl_index,
                        "local_shard_id": local_shard_id,
                    })

                    samples = []
                    gc.collect()

        # 保存最后不足 shard_size 的样本
        if len(samples) > 0:
            local_shard_id += 1
            shard_name_id = jsonl_index * 1000000 + local_shard_id

            shard_path = save_shard(
                samples=samples,
                output_dir=args.output_dir,
                shard_global_id=shard_name_id,
                prefix=args.output_prefix,
            )

            result["shards"].append({
                "path": shard_path,
                "num_samples": len(samples),
                "jsonl_index": jsonl_index,
                "local_shard_id": local_shard_id,
            })

            samples = []
            gc.collect()

        result["label_counter"] = dict(label_counter)

    except Exception as e:
        result["error"] = repr(e)

    finally:
        samples = []
        gc.collect()

    return result


def merge_label_counters(results):
    counter = Counter()

    for r in results:
        for k, v in r.get("label_counter", {}).items():
            counter[k] += v

    return dict(counter)


def write_manifest_and_info(
    results,
    args,
    label_map,
    special_ids,
    vocab_size,
):
    os.makedirs(args.output_dir, exist_ok=True)

    # 按 jsonl 原始顺序 + shard 顺序排序
    results = sorted(results, key=lambda x: x["jsonl_index"])

    all_shards = []

    for r in results:
        shards = r.get("shards", [])
        shards = sorted(shards, key=lambda x: (x["jsonl_index"], x["local_shard_id"]))
        all_shards.extend(shards)

    manifest_path = args.manifest_path
    if manifest_path is None:
        manifest_path = os.path.join(args.output_dir, "pt_manifest.jsonl")

    with open(manifest_path, "w", encoding="utf-8") as writer:
        for idx, shard in enumerate(all_shards, start=1):
            record = {
                "shard_index": idx,
                "path": shard["path"],
                "num_samples": shard["num_samples"],
                "jsonl_index": shard["jsonl_index"],
                "local_shard_id": shard["local_shard_id"],
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    label_map_path = args.label_map_path
    if label_map_path is None:
        label_map_path = os.path.join(args.output_dir, "label_map.json")

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    dataset_info_path = args.dataset_info_path
    if dataset_info_path is None:
        dataset_info_path = os.path.join(args.output_dir, "dataset_info.json")

    total_lines = sum(r["total_lines"] for r in results)
    valid_samples = sum(r["valid_samples"] for r in results)
    skipped_samples = sum(r["skipped_samples"] for r in results)
    total_errors = sum(1 for r in results if r["error"] is not None)
    label_counter = merge_label_counters(results)

    bpe_tokenizer_model_type = None
    if getattr(args, "tokenizer_type", "hex") == "bpe":
        bpe_tokenizer_model_type = infer_bpe_tokenizer_model_type(
            getattr(args, "bpe_tokenizer_path", None)
        )

    dataset_info = {
        "input": args.input,
        "vocab_path": args.vocab_path,
        "tokenizer_type": getattr(args, "tokenizer_type", "hex"),
        "bpe_tokenizer_path": getattr(args, "bpe_tokenizer_path", None),
        "bpe_tokenizer_model": getattr(args, "bpe_tokenizer_model", None),
        "bpe_tokenizer_model_type": bpe_tokenizer_model_type,
        "jsonl_max_burst_len": getattr(args, "jsonl_max_burst_len", None),
        "output_dir": args.output_dir,
        "manifest_path": manifest_path,
        "label_map_path": label_map_path,
        "max_burst_num": args.max_burst_num,
        "max_burst_len": args.max_burst_len,
        "shard_size": args.shard_size,
        "num_workers": args.num_workers,
        "num_shards": len(all_shards),
        "num_samples": valid_samples,
        "total_json_lines": total_lines,
        "skipped_samples": skipped_samples,
        "errors": total_errors,
        "vocab_size": vocab_size,
        "special_ids": special_ids,
        "label_map": label_map,
        "label_counter_by_id": label_counter,
        "fields": {
            "src": "[max_burst_num, max_burst_len]",
            "direction": "[max_burst_num], 0=dir0, 1=dir1, 2=padding",
            "burst_mask": "[max_burst_num], 1=real burst, 0=padding",
            "token_mask": "[max_burst_num, max_burst_len], 1=real token, 0=padding",
            "label": "scalar, -1 means unknown/unlabeled",
        }
    }

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print(f"jsonl files: {len(results)}")
    print(f"errors: {total_errors}")
    print(f"valid samples: {valid_samples}")
    print(f"skipped samples: {skipped_samples}")
    print(f"num shards: {len(all_shards)}")
    print(f"manifest: {manifest_path}")
    print(f"label map: {label_map_path}")
    print(f"dataset info: {dataset_info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert flow_burst JSONL files to sharded hierarchical PyTorch datasets."
    )

    parser.add_argument("--input", type=str, default="C:/HierarchicalBurst-FlowBERT/datasets/pre_train_output_jsonl/jsonl_parts", help="输入 flow_burst.jsonl 或 jsonl_parts 目录")
    parser.add_argument("--vocab_path", type=str, default="C:/HierarchicalBurst-FlowBERT/datasets/pre_train_output_jsonl/hex_vocab.txt", help="hex_vocab.txt 路径")
    parser.add_argument("--tokenizer_type", choices=["hex", "bpe"], default="hex", help="token id 生成方式")
    parser.add_argument("--bpe_tokenizer_path", type=str, default=None, help="tokenizer_type=bpe 时使用的 bpe_tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="C:/HierarchicalBurst-FlowBERT/datasets/pretrain_data", help="输出 pt 分片目录")

    parser.add_argument("--output_prefix", type=str, default="hierarchical", help="pt 分片文件名前缀")
    parser.add_argument("--manifest_path", type=str, default=None, help="输出 pt_manifest.jsonl 路径")
    parser.add_argument("--label_map_path", type=str, default=None, help="输出/读取 label_map.json 路径")
    parser.add_argument("--dataset_info_path", type=str, default=None, help="输出 dataset_info.json 路径")

    parser.add_argument("--max_burst_num", type=int, default=24, help="每个 flow 最多保留多少个 burst")
    parser.add_argument("--max_burst_len", type=int, default=128, help="每个 burst 最多保留多少个 token id，包括 [CLS]/[SEP]")

    parser.add_argument("--shard_size", type=int, default=50000, help="每个 .pt 分片保存多少条样本")
    parser.add_argument("--num_workers", type=int, default=12, help="多进程数量")

    parser.add_argument("--no_lowercase", action="store_true", help="不将 hex token 转小写")
    parser.add_argument("--lowercase_label", action="store_true", help="是否将 label 转小写")
    parser.add_argument("--no_cls_sep", action="store_true", help="不在每个 burst 前后加 [CLS]/[SEP]")
    parser.add_argument("--keep_meta", action="store_true", help="是否在 pt 样本中保留 flow_id、pcap、label_name 等字符串元信息")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    jsonl_files = list(iter_jsonl_files(args.input))

    if len(jsonl_files) == 0:
        raise FileNotFoundError(f"没有找到 jsonl 文件: {args.input}")

    # 主进程先加载一次 vocab，用于写 dataset_info；worker 里会各自加载一份
    vocab, special_ids = load_vocab(
        vocab_path=args.vocab_path,
        lowercase=not args.no_lowercase,
    )

    if args.label_map_path is None:
        args.label_map_path = os.path.join(args.output_dir, "label_map.json")

    label_map = build_or_load_label_map(
        input_path=args.input,
        label_map_path=args.label_map_path if os.path.exists(args.label_map_path) else None,
        lowercase_label=args.lowercase_label,
    )

    args_dict = vars(args).copy()

    tasks = []
    for idx, jsonl_path in enumerate(jsonl_files, start=1):
        tasks.append({
            "jsonl_index": idx,
            "jsonl_path": jsonl_path,
            "args_dict": args_dict,
        })

    print("\n[START]")
    print(f"jsonl files: {len(tasks)}")
    print(f"num_workers: {args.num_workers}")
    print(f"shard_size: {args.shard_size}")
    print(f"output_dir: {args.output_dir}")

    results = []

    if args.num_workers <= 1:
        init_worker(
            vocab_path=args.vocab_path,
            label_map=label_map,
            lowercase=not args.no_lowercase,
            tokenizer_type=args.tokenizer_type,
            bpe_tokenizer_path=args.bpe_tokenizer_path,
        )

        for task in tasks:
            print(f"[JSONL {task['jsonl_index']}/{len(tasks)}] {task['jsonl_path']}")
            result = process_jsonl_file(task)
            results.append(result)

            if result["error"] is not None:
                print(f"  ERROR: {result['error']}")
            else:
                print(
                    f"  lines={result['total_lines']}, "
                    f"valid={result['valid_samples']}, "
                    f"skipped={result['skipped_samples']}, "
                    f"shards={len(result['shards'])}"
                )
    else:
        ctx = mp.get_context("spawn")

        with ctx.Pool(
            processes=args.num_workers,
            initializer=init_worker,
            initargs=(
                args.vocab_path,
                label_map,
                not args.no_lowercase,
                args.tokenizer_type,
                args.bpe_tokenizer_path,
            ),
        ) as pool:
            for result in pool.imap_unordered(process_jsonl_file, tasks):
                results.append(result)

                idx = result["jsonl_index"]

                if result["error"] is not None:
                    print(f"[JSONL {idx}/{len(tasks)}] ERROR: {result['error']}")
                else:
                    print(
                        f"[JSONL {idx}/{len(tasks)}] done | "
                        f"lines={result['total_lines']}, "
                        f"valid={result['valid_samples']}, "
                        f"skipped={result['skipped_samples']}, "
                        f"shards={len(result['shards'])}"
                    )

    write_manifest_and_info(
        results=results,
        args=args,
        label_map=label_map,
        special_ids=special_ids,
        vocab_size=len(vocab),
    )


if __name__ == "__main__":
    main()
