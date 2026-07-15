# run_all_pipeline.py

import os
import json
import argparse
import multiprocessing as mp
from types import SimpleNamespace


from pcap_to_flow_burst_jsonl_zero_ip_port import write_flow_burst_jsonl

from build_hex_vocab_from_jsonl import build_vocab

from jsonl_to_hierarchical_pt_sharded import (
    iter_jsonl_files,
    load_vocab,
    build_or_load_label_map,
    init_worker,
    process_jsonl_file,
    write_manifest_and_info,
)

from merge_split_finetune_pt import (
    find_pt_files,
    load_all_samples,
    stratified_split,
    count_labels,
    save_pt,
    inspect_first_sample,
)


# ============================================================
# STEP 1: PCAP -> JSONL
# ============================================================

def run_pcap_to_jsonl(args):
    """
    第一步：
    调用 pcap_to_jsonl.py 文件中的 write_flow_burst_jsonl(args)
    将 pcap / pcapng 转成 flow_burst JSONL。
    """

    pcap_args = SimpleNamespace(
        input=args.pcap_input,
        output_jsonl=args.output_jsonl,

        token_bytes=args.token_bytes,
        max_burst_num=args.max_burst_num,
        max_burst_len=args.max_burst_len,
        min_burst_len=args.min_burst_len,
        min_burst_num=args.min_burst_num,
        max_packet_tokens=args.max_packet_tokens,
        max_packets_per_flow=args.max_packets_per_flow,
        tcp_idle_timeout=args.tcp_idle_timeout,
        tcp_handshake_timeout=args.tcp_handshake_timeout,
        tcp_close_timeout=args.tcp_close_timeout,
        udp_idle_timeout=args.udp_idle_timeout,
        cut_link_layer=args.cut_link_layer,

        label_mode=args.label_mode,
        label_csv=args.label_csv,
        csv_encoding=args.csv_encoding,
        csv_src_ip_col=args.csv_src_ip_col,
        csv_src_port_col=args.csv_src_port_col,
        csv_dst_ip_col=args.csv_dst_ip_col,
        csv_dst_port_col=args.csv_dst_port_col,
        csv_proto_col=args.csv_proto_col,
        csv_label_col=args.csv_label_col,
        drop_unmatched_csv=args.drop_unmatched_csv,

        num_workers=args.pcap_num_workers,
        temp_dir=args.jsonl_parts_dir,
        merge_parts=args.merge_parts,
        keep_parts=args.keep_parts,
    )

    print("\n========== STEP 1: PCAP -> JSONL ==========")
    write_flow_burst_jsonl(pcap_args)


# ============================================================
# STEP 2: JSONL -> VOCAB
# ============================================================

def run_build_vocab(args):
    """
    第二步：
    调用 build_vocab.py 文件中的 build_vocab(...)
    根据 jsonl_parts 构建 hex_vocab.txt。
    """

    print("\n========== STEP 2: JSONL -> VOCAB ==========")

    build_vocab(
        input_path=args.vocab_input,
        output_vocab=args.output_vocab,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        lowercase=not args.no_lowercase,
    )


# ============================================================
# STEP 3: JSONL -> PT SHARDS
# ============================================================

def run_jsonl_to_pt(args):
    """
    第三步：
    调用 jsonl_to_pt.py 文件中的函数，
    将 JSONL + vocab 转成 PyTorch .pt 分片。
    """

    print("\n========== STEP 3: JSONL -> PT SHARDS ==========")

    os.makedirs(args.pt_output_dir, exist_ok=True)

    jsonl_files = list(iter_jsonl_files(args.pt_input))

    if len(jsonl_files) == 0:
        raise FileNotFoundError(f"没有找到 jsonl 文件: {args.pt_input}")

    vocab, special_ids = load_vocab(
        vocab_path=args.output_vocab,
        lowercase=not args.no_lowercase,
    )

    label_map_path = args.label_map_path

    label_map = build_or_load_label_map(
        input_path=args.pt_input,
        label_map_path=label_map_path if os.path.exists(label_map_path) else None,
        lowercase_label=args.lowercase_label,
    )

    pt_args = SimpleNamespace(
        input=args.pt_input,
        vocab_path=args.output_vocab,
        tokenizer_type=args.tokenizer_type,
        bpe_tokenizer_path=args.bpe_tokenizer_path,
        jsonl_max_burst_len=args.max_burst_len,
        output_dir=args.pt_output_dir,

        output_prefix=args.output_prefix,
        manifest_path=args.manifest_path,
        label_map_path=args.label_map_path,
        dataset_info_path=args.dataset_info_path,

        max_burst_num=args.pt_max_burst_num,
        max_burst_len=args.pt_max_burst_len,

        shard_size=args.shard_size,
        num_workers=args.pt_num_workers,

        no_lowercase=args.no_lowercase,
        lowercase_label=args.lowercase_label,
        no_cls_sep=args.no_cls_sep,
        keep_meta=args.keep_meta,
    )

    args_dict = vars(pt_args).copy()

    tasks = []

    for idx, jsonl_path in enumerate(jsonl_files, start=1):
        tasks.append({
            "jsonl_index": idx,
            "jsonl_path": jsonl_path,
            "args_dict": args_dict,
        })

    print("\n[START PT CONVERT]")
    print(f"jsonl files: {len(tasks)}")
    print(f"num_workers: {pt_args.num_workers}")
    print(f"shard_size: {pt_args.shard_size}")
    print(f"output_dir: {pt_args.output_dir}")

    results = []

    if pt_args.num_workers <= 1:
        init_worker(
            vocab_path=pt_args.vocab_path,
            label_map=label_map,
            lowercase=not pt_args.no_lowercase,
            tokenizer_type=pt_args.tokenizer_type,
            bpe_tokenizer_path=pt_args.bpe_tokenizer_path,
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
            processes=pt_args.num_workers,
            initializer=init_worker,
            initargs=(
                pt_args.vocab_path,
                label_map,
                not pt_args.no_lowercase,
                pt_args.tokenizer_type,
                pt_args.bpe_tokenizer_path,
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
        args=pt_args,
        label_map=label_map,
        special_ids=special_ids,
        vocab_size=len(vocab),
    )


# ============================================================
# STEP 4: PT SHARDS -> TRAIN / VALID / TEST
# ============================================================

def run_split_pt_dataset(args):
    """
    第四步：
    调用 split_pt_dataset.py 文件中的函数，
    将 pt 分片划分为 train.pt / valid.pt / test.pt。

    注意：
    这里没有复制拆分文件的全部逻辑，
    而是调用你文件中的这些函数：
    find_pt_files
    load_all_samples
    stratified_split
    count_labels
    save_pt
    inspect_first_sample
    """

    print("\n========== STEP 4: PT SHARDS -> TRAIN / VALID / TEST ==========")

    os.makedirs(args.split_output_dir, exist_ok=True)

    pt_files = find_pt_files(args.split_input_dir)

    if len(pt_files) == 0:
        raise ValueError(f"没有在目录中找到 .pt 文件: {args.split_input_dir}")

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

    train_path = os.path.join(args.split_output_dir, "train.pt")
    valid_path = os.path.join(args.split_output_dir, "valid.pt")
    test_path = os.path.join(args.split_output_dir, "test.pt")

    save_pt(train_set, train_path)
    save_pt(valid_set, valid_path)
    save_pt(test_set, test_path)

    split_info = {
        "input_dir": args.split_input_dir,
        "output_dir": args.split_output_dir,
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

    split_info_path = os.path.join(args.split_output_dir, "split_info.json")

    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(f"[SAVE INFO] {split_info_path}")

    print("\n[SPLIT DONE]")
    print("train:", len(train_set), count_labels(train_set))
    print("valid:", len(valid_set), count_labels(valid_set))
    print("test :", len(test_set), count_labels(test_set))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run full data pipeline: PCAP -> JSONL -> VOCAB -> PT -> SPLIT"
    )

    # ============================================================
    # 总控开关
    # ============================================================

    parser.add_argument(
        "--skip_pcap_to_jsonl",
        action="store_true",
        help="跳过 PCAP 转 JSONL",
    )

    parser.add_argument(
        "--skip_build_vocab",
        action="store_true",
        help="跳过构建 vocab",
    )

    parser.add_argument(
        "--skip_jsonl_to_pt",
        action="store_true",
        help="跳过 JSONL 转 PT",
    )

    parser.add_argument(
        "--skip_split_pt",
        action="store_true",
        help="跳过 PT 分片划分 train/valid/test",
    )

    # ============================================================
    # 输入路径
    # ============================================================

    parser.add_argument(
        "--pcap_input",
        type=str,
        default="C:/data/fine-tuning/cstnet-tls",
        help="输入 pcap / pcapng 文件或目录",
    )

    # ============================================================
    # 统一输出根目录
    # ============================================================

    parser.add_argument(
        "--output_root",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/cstnet_tls_only_de_ip_pipeline",
        help="统一输出根目录，jsonl/vocab/pt/split/logs 都会放在这里下面",
    )

    # ============================================================
    # PCAP -> JSONL 参数
    # ============================================================

    parser.add_argument(
        "--token_bytes",
        type=int,
        default=2,
        help="每个 token 包含几个字节，默认 2，即 4 个 hex 字符",
    )

    parser.add_argument(
        "--max_burst_num",
        type=int,
        default=16,
        help="PCAP 转 JSONL 时，每个 flow 最多保留多少个 burst",
    )

    parser.add_argument(
        "--max_burst_len",
        type=int,
        default=128,
        help="PCAP 转 JSONL 时，每个 burst 最多保留多少个 token",
    )

    parser.add_argument(
        "--min_burst_len",
        type=int,
        default=1,
        help="过滤 token 数小于该值的 burst",
    )

    parser.add_argument(
        "--min_burst_num",
        type=int,
        default=1,
        help="过滤 burst 数小于该值的 flow",
    )

    parser.add_argument(
        "--max_packet_tokens",
        type=int,
        default=None,
        help="每个 packet 最多保留多少 token，默认不限制",
    )

    parser.add_argument(
        "--max_packets_per_flow",
        type=int,
        default=None,
        help="每个 flow 最多保留多少 packet，默认不限制",
    )

    parser.add_argument("--tcp_idle_timeout", type=float, default=120.0)
    parser.add_argument("--tcp_handshake_timeout", type=float, default=30.0)
    parser.add_argument("--tcp_close_timeout", type=float, default=15.0)
    parser.add_argument("--udp_idle_timeout", type=float, default=30.0)

    parser.add_argument(
        "--cut_link_layer",
        action="store_true",
        default=True,
        help="是否从 IP 层开始取 bytes，默认开启",
    )

    parser.add_argument(
        "--label_mode",
        type=str,
        default="parent",
        choices=["unknown", "parent", "csv"],
        help="标签模式：unknown 无标签；parent 使用上级目录名；csv 根据 CSV 五元组匹配",
    )

    parser.add_argument(
        "--label_csv",
        type=str,
        default=None,
        help="标签 CSV 文件路径，仅 label_mode=csv 时需要",
    )

    parser.add_argument(
        "--csv_encoding",
        type=str,
        default="utf-8-sig",
        help="CSV 文件编码",
    )

    parser.add_argument(
        "--csv_src_ip_col",
        type=str,
        default="Source IP",
        help="CSV 中源 IP 列名",
    )

    parser.add_argument(
        "--csv_src_port_col",
        type=str,
        default="Source Port",
        help="CSV 中源端口列名",
    )

    parser.add_argument(
        "--csv_dst_ip_col",
        type=str,
        default="Destination IP",
        help="CSV 中目的 IP 列名",
    )

    parser.add_argument(
        "--csv_dst_port_col",
        type=str,
        default="Destination Port",
        help="CSV 中目的端口列名",
    )

    parser.add_argument(
        "--csv_proto_col",
        type=str,
        default="Protocol",
        help="CSV 中协议列名",
    )

    parser.add_argument(
        "--csv_label_col",
        type=str,
        default="Label",
        help="CSV 中标签列名",
    )

    parser.add_argument(
        "--drop_unmatched_csv",
        action="store_true",
        help="label_mode=csv 时，是否丢弃 CSV 匹配不到标签的 flow",
    )

    parser.add_argument(
        "--pcap_num_workers",
        type=int,
        default=16,
        help="PCAP 转 JSONL 的进程数。Windows 建议先用 1",
    )

    parser.add_argument(
        "--merge_parts",
        action="store_true",
        default=True,
        help="是否合并 jsonl_parts 为 flow_burst_labeled.jsonl",
    )

    parser.add_argument(
        "--keep_parts",
        action="store_true",
        default=True,
        help="合并后是否保留 jsonl_parts",
    )

    # ============================================================
    # VOCAB 参数
    # ============================================================

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
        help="最大词表大小，包含特殊符号；None 表示不限制",
    )

    parser.add_argument(
        "--no_lowercase",
        action="store_true",
        help="不将 hex token 转为小写",
    )

    parser.add_argument(
        "--tokenizer_type",
        choices=["hex", "bpe"],
        default="hex",
        help="JSONL -> PT 时的 token id 生成方式。BPE 时需跳过重建词表并提供 bpe_tokenizer_path",
    )

    parser.add_argument(
        "--bpe_tokenizer_path",
        type=str,
        default=None,
        help="tokenizer_type=bpe 时使用的预训练 bpe_tokenizer.json",
    )

    # ============================================================
    # JSONL -> PT 参数
    # ============================================================

    parser.add_argument(
        "--pt_max_burst_num",
        type=int,
        default=24,
        help="转 pt 时每个 flow 最多保留多少个 burst",
    )

    parser.add_argument(
        "--pt_max_burst_len",
        type=int,
        default=128,
        help="转 pt 时每个 burst 最多保留多少 token id，包括 [CLS]/[SEP]",
    )

    parser.add_argument(
        "--shard_size",
        type=int,
        default=50000,
        help="每个 .pt 分片保存多少条样本",
    )

    parser.add_argument(
        "--pt_num_workers",
        type=int,
        default=24,
        help="JSONL 转 PT 的进程数。Windows 建议先用 1",
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        default="hierarchical",
        help=".pt 分片文件名前缀",
    )

    parser.add_argument(
        "--lowercase_label",
        action="store_true",
        help="是否将 label 转成小写",
    )

    parser.add_argument(
        "--no_cls_sep",
        action="store_true",
        help="不在每个 burst 前后添加 [CLS] 和 [SEP]",
    )

    parser.add_argument(
        "--keep_meta",
        action="store_true",
        help="是否在 pt 样本中保留 flow_id、pcap、label_name 等字符串信息",
    )

    # ============================================================
    # PT SPLIT 参数
    # ============================================================

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例",
    )

    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="验证集比例",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="测试集比例",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="划分训练/验证/测试集的随机种子",
    )

    args = parser.parse_args()

    # ============================================================
    # 检查比例
    # ============================================================

    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + valid_ratio + test_ratio 必须等于 1，"
            f"当前为 {ratio_sum}"
        )

    # ============================================================
    # 自动生成统一输出目录结构
    # ============================================================

    args.output_root = os.path.abspath(args.output_root)

    args.jsonl_dir = os.path.join(args.output_root, "jsonl")
    args.vocab_dir = os.path.join(args.output_root, "vocab")
    args.pt_output_dir = os.path.join(args.output_root, "pt")
    args.split_output_dir = os.path.join(args.output_root, "split")
    args.logs_dir = os.path.join(args.output_root, "logs")

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.jsonl_dir, exist_ok=True)
    os.makedirs(args.vocab_dir, exist_ok=True)
    os.makedirs(args.pt_output_dir, exist_ok=True)
    os.makedirs(args.split_output_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    # ============================================================
    # 自动生成各阶段输出路径
    # ============================================================

    # JSONL 输出
    args.output_jsonl = os.path.join(
        args.jsonl_dir,
        "flow_burst_labeled.jsonl"
    )

    args.jsonl_parts_dir = os.path.join(
        args.jsonl_dir,
        "jsonl_parts"
    )

    # vocab 输出
    args.output_vocab = os.path.join(
        args.vocab_dir,
        "hex_vocab.txt"
    )

    # PT 输出相关文件
    args.manifest_path = os.path.join(
        args.pt_output_dir,
        "pt_manifest.jsonl"
    )

    args.label_map_path = os.path.join(
        args.pt_output_dir,
        "label_map.json"
    )

    args.dataset_info_path = os.path.join(
        args.pt_output_dir,
        "dataset_info.json"
    )

    # build_vocab 和 jsonl_to_pt 默认都读取 jsonl_parts
    args.vocab_input = args.jsonl_parts_dir
    args.pt_input = args.jsonl_parts_dir

    # split 阶段默认读取 pt 分片目录，输出到 split 目录
    args.split_input_dir = args.pt_output_dir

    # ============================================================
    # 打印配置
    # ============================================================

    print("\n========== PIPELINE CONFIG ==========")
    print(f"pcap_input:        {args.pcap_input}")
    print(f"output_root:       {args.output_root}")

    print("\n[JSONL]")
    print(f"jsonl_dir:         {args.jsonl_dir}")
    print(f"jsonl_parts_dir:   {args.jsonl_parts_dir}")
    print(f"output_jsonl:      {args.output_jsonl}")

    print("\n[VOCAB]")
    print(f"vocab_dir:         {args.vocab_dir}")
    print(f"output_vocab:      {args.output_vocab}")
    print(f"tokenizer_type:    {args.tokenizer_type}")
    print(f"bpe_tokenizer:     {args.bpe_tokenizer_path}")

    print("\n[PT SHARDS]")
    print(f"pt_output_dir:     {args.pt_output_dir}")
    print(f"manifest_path:     {args.manifest_path}")
    print(f"label_map_path:    {args.label_map_path}")
    print(f"dataset_info_path: {args.dataset_info_path}")

    print("\n[SPLIT]")
    print(f"split_input_dir:   {args.split_input_dir}")
    print(f"split_output_dir:  {args.split_output_dir}")
    print(f"train_ratio:       {args.train_ratio}")
    print(f"valid_ratio:       {args.valid_ratio}")
    print(f"test_ratio:        {args.test_ratio}")
    print(f"seed:              {args.seed}")
    print("====================================")

    if args.tokenizer_type == "bpe" and not args.skip_build_vocab:
        raise ValueError(
            "tokenizer_type=bpe 时旧微调入口不会训练 BPE；"
            "请使用 --skip_build_vocab，并让 --output_vocab 指向预训练 bpe_vocab.txt。"
        )

    # ============================================================
    # 执行四个阶段
    # ============================================================

    if not args.skip_pcap_to_jsonl:
        run_pcap_to_jsonl(args)
    else:
        print("\n[SKIP] PCAP -> JSONL")

    if not args.skip_build_vocab:
        run_build_vocab(args)
    else:
        print("\n[SKIP] JSONL -> VOCAB")

    if not args.skip_jsonl_to_pt:
        run_jsonl_to_pt(args)
    else:
        print("\n[SKIP] JSONL -> PT SHARDS")

    if not args.skip_split_pt:
        run_split_pt_dataset(args)
    else:
        print("\n[SKIP] PT SHARDS -> TRAIN / VALID / TEST")

    print("\n========== ALL DONE ==========")
    print(f"JSONL output:      {args.output_jsonl}")
    print(f"JSONL parts:       {args.jsonl_parts_dir}")
    print(f"VOCAB output:      {args.output_vocab}")
    print(f"PT shards:         {args.pt_output_dir}")
    print(f"SPLIT output:      {args.split_output_dir}")
    print(f"train.pt:          {os.path.join(args.split_output_dir, 'train.pt')}")
    print(f"valid.pt:          {os.path.join(args.split_output_dir, 'valid.pt')}")
    print(f"test.pt:           {os.path.join(args.split_output_dir, 'test.pt')}")
    print(f"split_info.json:   {os.path.join(args.split_output_dir, 'split_info.json')}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
