import argparse
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch

from convert_checkpoint_to_bin import extract_model_state_dict, remove_module_prefix
from process_pretrain_data.build_hex_vocab_from_jsonl import build_vocab
from process_pretrain_data.pcap_to_flow_burst_json import write_flow_burst_jsonl as write_pretrain_jsonl
from process_fine_tuning.pcap_to_flow_burst_jsonl_zero_ip_port import (
    write_flow_burst_jsonl as write_finetune_jsonl,
)
from process_fine_tuning.merge_split_finetune_pt import (
    count_labels,
    find_pt_files,
    inspect_first_sample,
    load_all_samples,
    save_pt,
    stratified_split,
)


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

PARAMETER_DESCRIPTIONS = {
    "common": {
        "pcap_input": "输入 PCAP/PCAPNG 文件或文件夹。文件夹名会进入 run_name，方便区分数据集。",
        "output_root": "实验输出根目录。每次运行会在其下创建带时间、数据集名和关键参数的子目录。",
        "run_name": "可选实验名。不传时自动使用时间、数据集名、学习率、burst/token 数、batch 等生成。",
        "keep_intermediate": "是否保留中间文件。默认会清理 JSONL parts、微调临时 PT 分片等过程文件。",
    },
    "flow": {
        "tokenizer": "token id 生成方式：hex 表示固定长度 hex token；bpe 表示训练并复用 tokenizer。",
        "token_bytes": "每个 token 包含多少字节。默认 2，表示 4 个 hex 字符一个 token。",
        "jsonl_max_burst_len": "PCAP->JSONL 阶段每个 burst 保留多少固定 hex token。None 表示自动选择；BPE 时建议大于最终 max_burst_len。",
        "max_burst_num": "每个 flow 最多保留多少个 burst。超过会截断；微调阶段会自动使用预训练 dataset_info 中的值。",
        "max_burst_len": "每个 burst 最多保留多少 token id。默认会包含 [CLS]/[SEP]，实际内容长度通常为 max_burst_len - 2。",
        "min_burst_len": "过滤过短 burst。token 数少于该值的 burst 会被丢弃。",
        "min_burst_num": "过滤过短 flow。burst 数少于该值的 flow 会被丢弃。",
        "max_packet_tokens": "单个 packet 最多保留多少 token。None 表示不限制。",
        "max_packets_per_flow": "单个 flow 最多保留多少 packet。None 表示不限制。",
        "tcp_idle_timeout": "TCP 已建立/中途会话的空闲切分秒数；设为 0 可关闭超时兜底。",
        "tcp_handshake_timeout": "TCP 半连接等待秒数，超时后结束当前会话。",
        "tcp_close_timeout": "看到 FIN 后等待完整关闭握手的秒数。",
        "udp_idle_timeout": "UDP 双向五元组的空闲切分秒数。",
        "cut_link_layer": "是否从 IP 层开始取 bytes，去掉链路层头。CLI 中用 --no_cut_link_layer 关闭。",
        "label_mode": "标签来源：unknown 表示无标签；parent 表示使用 pcap 上级目录名；csv 表示根据 CSV 五元组匹配。",
        "label_csv": "label_mode=csv 时使用的标签 CSV 路径。",
        "drop_unmatched_csv": "label_mode=csv 时，是否丢弃 CSV 中匹配不到标签的 flow。",
        "pcap_num_workers": "PCAP 转 JSONL 的进程数。追求速度时可设为 32。",
        "pcap_progress_every": "PCAP 转 JSONL 时每处理多少个 pcap 打印一次进度；0 表示只打印错误和汇总。",
        "pcap_chunksize": "PCAP 多进程任务批量分发大小。小 pcap 很多时可设为 16/32/64 减少调度开销。",
        "pcap_maxtasksperchild": "每个 PCAP worker 处理多少个任务后重启。None 表示不重启；小 pcap 很多时不要设为 1。",
        "shuffle_data": "是否在数据准备阶段打乱样本文件顺序。默认开启，会打乱 PCAP 文件顺序和 JSONL 转 PT 的文件顺序。",
        "shuffle_seed": "数据准备阶段打乱顺序使用的随机种子。",
        "jsonl_to_pt_workers": "JSONL 转 PT 的进程数。追求速度时可设为 32。",
        "jsonl_progress_every": "JSONL 转 PT 时每处理多少个 JSONL 打印一次进度；0 表示只打印错误和汇总。",
        "shard_size": "每个 .pt 分片保存多少条样本。",
        "min_freq": "构建词表时，低于该频率的 token 不进入词表。主要用于预训练。",
        "max_vocab_size": "最大词表大小，包含特殊符号。None 表示不限制。",
        "bpe_vocab_size": "tokenizer=bpe 时的词表大小，ET-BERT WordPiece 固定为 65541。",
        "bpe_min_frequency": "旧 BPE merge 的最小频次；ET-BERT WordPiece 会记录该值但不做 merge。",
        "bpe_tokenizer_model": "tokenizer=bpe 时的实际 tokenizer：etbert_wordpiece 表示固定 0000-ffff 的 ET-BERT 风格 WordPiece；etbert_wordpiece_trainer 表示调用 WordPieceTrainer；bpe 表示旧的连续 hex BPE。",
        "bpe_unit": "tokenizer 训练语料单位：burst 表示每个 burst 一行；flow 表示每个 flow 一行且 burst 间用空格隔开。",
        "bpe_raw_burst_len_multiplier": "tokenizer=bpe 且 jsonl_max_burst_len=None 时，PCAP->JSONL 的固定 hex token 上限=max_burst_len*该倍率。",
        "no_lowercase": "是否不把 hex token 转小写。默认会转小写。",
        "lowercase_label": "是否把 label 转小写后再建立 label_map。",
        "no_cls_sep": "是否不在每个 burst 前后添加 [CLS] 和 [SEP]。默认会添加。",
        "keep_meta": "是否在 PT 样本中保留 flow_id、pcap、label_name 等字符串元信息。",
    },
    "model": {
        "hidden_size": "Transformer hidden size。",
        "burst_layers": "burst-level Transformer 层数。",
        "flow_layers": "flow-level Transformer 层数。",
        "num_heads": "多头注意力 head 数。",
        "intermediate_size": "Transformer FFN 中间层大小。",
        "dropout": "dropout 比例。",
        "no_direction_embedding": "是否关闭方向 embedding。默认使用方向 embedding。",
    },
    "pretrain": {
        "epochs": "预训练 epoch 数。通常与 max_steps 配合，达到 max_steps 后提前停止。",
        "batch_size": "每张卡/每个进程的 micro batch size。",
        "gradient_accumulation_steps": "梯度累积步数。有效全局 batch=batch_size*nproc_per_node*gradient_accumulation_steps。",
        "lr": "学习率。",
        "weight_decay": "AdamW weight decay。",
        "max_grad_norm": "梯度裁剪阈值。小于等于 0 表示不裁剪。",
        "drop_last": "DataLoader 是否丢弃最后一个不完整 batch。",
        "num_workers": "预训练 DataLoader worker 数。",
        "device": "运行设备。单进程可用 auto/cuda/cuda:0/cpu；DDP 通常保持 auto。",
        "cuda_devices": "指定可见 CUDA，例如 0 或 0,1,2,3。",
        "nproc_per_node": "DDP 进程数。多卡时通常等于 GPU 数量。",
        "master_port": "DDP 主进程监听端口。默认 29500；端口占用时改成 29501/29511 等。",
        "dist_backend": "DDP backend：auto/nccl/gloo。",
        "ddp_timeout_minutes": "DDP 初始化和通信超时时间。",
        "find_unused_parameters": "DDP 是否查找未使用参数。多任务头可能未全部使用，默认开启更稳。",
        "shuffle": "预训练 DataLoader 是否打乱数据。",
        "verbose_ranks": "是否打印所有 DDP rank 的调试日志。默认只打印 rank0。",
        "fp16": "是否使用混合精度。CLI 中用 --no_fp16 关闭。",
        "seed": "随机种子。",
        "log_every": "每多少个 optimizer update step 打印一次日志。",
        "save_every": "每多少个 optimizer update step 保存一次 checkpoint。",
        "max_steps": "最大 optimizer update step。注意不是 micro step。",
        "plot_loss": "是否保存 loss_log.csv 和 loss 曲线图。",
        "token_mlm_probability": "token-level MBM mask 概率。",
        "burst_mbm_probability": "burst-level MBM mask 概率。",
        "token_loss_weight": "token-level MBM loss 权重。",
        "burst_loss_weight": "burst-level MBM loss 权重。",
        "relation_task": "关系预训练任务：none/nsp/sodf/both。",
        "relation_pairs": "每个 batch 采样多少个关系任务 pair。",
        "nsp_positive_ratio": "NSP 正样本比例。",
        "nsp_loss_weight": "NSP loss 权重。",
        "sodf_loss_weight": "SODF loss 权重。",
    },
    "finetune": {
        "pretrained_model_path": "预训练阶段输出的 *_pretrained_model.bin。",
        "vocab_path": "预训练阶段输出的 *_hex_vocab.txt 或 BPE 的 bpe_vocab.txt。",
        "dataset_info_path": "预训练阶段输出的 *_dataset_info.json。微调会从这里读取模型输入尺寸。",
        "epochs": "微调 epoch 数。",
        "batch_size": "微调 batch size。",
        "lr": "微调学习率。",
        "weight_decay": "AdamW weight decay。",
        "max_grad_norm": "梯度裁剪阈值。",
        "drop_last": "训练 DataLoader 是否丢弃最后一个不完整 batch。",
        "max_steps": "最大微调 step。-1 表示不限制。",
        "num_workers": "微调 DataLoader worker 数。",
        "device": "微调运行设备，例如 auto/cuda/cuda:0/cpu。",
        "fp16": "是否使用混合精度。CLI 中用 --no_fp16 关闭。",
        "seed": "随机种子。",
        "log_every": "每多少 step 打印一次训练日志。",
        "train_ratio": "训练集比例。",
        "valid_ratio": "验证集比例。",
        "test_ratio": "测试集比例。train/valid/test 三者之和必须为 1。",
    },
}


def _load_module(name, relative_path):
    path = os.path.join(PROJECT_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pretrain_data = importlib.import_module(
    "process_pretrain_data.jsonl_to_hierarchical_pt_sharded"
)
finetune_data = importlib.import_module(
    "process_fine_tuning.jsonl_to_hierarchical_pt_sharded"
)
pretrain_runner = _load_module(
    "pipeline_pretrain_runner",
    "pre-train/pretrain_hierarchical_ddp_grad_accum.py",
)
finetune_runner = _load_module(
    "pipeline_finetune_runner",
    "fine-tuning/finetune_hierarchical.py",
)


@dataclass
class FlowOptions:
    tokenizer: str = "hex"
    token_bytes: int = 2
    jsonl_max_burst_len: int | None = None
    max_burst_num: int = 24
    max_burst_len: int = 128
    min_burst_len: int = 1
    min_burst_num: int = 1
    max_packet_tokens: int | None = None
    max_packets_per_flow: int | None = None
    tcp_idle_timeout: float = 120.0
    tcp_handshake_timeout: float = 30.0
    tcp_close_timeout: float = 15.0
    udp_idle_timeout: float = 30.0
    cut_link_layer: bool = True
    label_mode: str = "unknown"
    label_csv: str | None = None
    csv_encoding: str = "utf-8-sig"
    csv_src_ip_col: str = "Source IP"
    csv_src_port_col: str = "Source Port"
    csv_dst_ip_col: str = "Destination IP"
    csv_dst_port_col: str = "Destination Port"
    csv_proto_col: str = "Protocol"
    csv_label_col: str = "Label"
    drop_unmatched_csv: bool = False
    pcap_num_workers: int = 1
    pcap_progress_every: int = 100
    pcap_chunksize: int = 16
    pcap_maxtasksperchild: int | None = None
    shuffle_data: bool = True
    shuffle_seed: int = 42
    jsonl_to_pt_workers: int = 1
    jsonl_progress_every: int = 50
    shard_size: int = 50000
    min_freq: int = 1
    max_vocab_size: int | None = None
    bpe_vocab_size: int = 65541
    bpe_min_frequency: int = 2
    bpe_tokenizer_model: str = "etbert_wordpiece"
    bpe_unit: str = "burst"
    bpe_raw_burst_len_multiplier: int = 2
    bpe_tokenizer_path: str | None = None
    no_lowercase: bool = False
    lowercase_label: bool = False
    no_cls_sep: bool = False
    keep_meta: bool = False


@dataclass
class ModelOptions:
    hidden_size: int = 256
    burst_layers: int = 4
    flow_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 1024
    dropout: float = 0.1
    no_direction_embedding: bool = False


@dataclass
class PretrainOptions:
    epochs: int = 10000
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    drop_last: bool = True
    num_workers: int = 1
    device: str = "auto"
    cuda_devices: str | None = None
    nproc_per_node: int = 1
    master_port: int = 29500
    dist_backend: str = "auto"
    ddp_timeout_minutes: int = 60
    find_unused_parameters: bool = True
    shuffle: bool = False
    verbose_ranks: bool = False
    fp16: bool = True
    seed: int = 42
    log_every: int = 5
    save_every: int = 2000
    max_steps: int = 80000
    plot_loss: bool = True
    token_mlm_probability: float = 0.15
    burst_mbm_probability: float = 0.15
    token_loss_weight: float = 0.1
    burst_loss_weight: float = 1.0
    relation_task: str = "sodf"
    relation_pairs: int = 32
    nsp_positive_ratio: float = 0.5
    nsp_loss_weight: float = 1.0
    sodf_loss_weight: float = 1.0


@dataclass
class FinetuneOptions:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    drop_last: bool = False
    max_steps: int = -1
    num_workers: int = 0
    device: str = "auto"
    fp16: bool = True
    seed: int = 42
    log_every: int = 10
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _safe_text(value):
    text = str(value)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in [".", "-", "_"]:
            keep.append(ch)
        else:
            keep.append("-")
    return "".join(keep).strip("-")


def _run_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _lr_text(lr):
    return _safe_text(f"{lr:.0e}" if lr < 0.001 else str(lr))


def _dataset_name(input_path):
    path = os.path.normpath(str(input_path))
    name = os.path.basename(path)
    if "." in name:
        stem, ext = os.path.splitext(name)
        if stem:
            name = stem
    return _safe_text(name or "dataset")


def _pretrain_run_name(flow, model, train, dataset_name, run_name=None):
    if run_name:
        return _safe_text(run_name)
    cuda_text = f"cuda{_safe_text(train.cuda_devices)}" if train.cuda_devices else "cudaauto"
    tokenizer_text = (
        f"{flow.tokenizer}-{flow.bpe_tokenizer_model}"
        if flow.tokenizer == "bpe"
        else flow.tokenizer
    )
    return "_".join([
        _run_stamp(),
        "pretrain",
        tokenizer_text,
        dataset_name,
        f"lr{_lr_text(train.lr)}",
        f"bn{flow.max_burst_num}",
        f"tl{flow.max_burst_len}",
        f"h{model.hidden_size}",
        f"bs{train.batch_size}",
        f"acc{train.gradient_accumulation_steps}",
        cuda_text,
        f"steps{train.max_steps}",
    ])


def _finetune_run_name(flow, model, train, dataset_name, run_name=None):
    if run_name:
        return _safe_text(run_name)
    tokenizer_text = (
        f"{flow.tokenizer}-{flow.bpe_tokenizer_model}"
        if flow.tokenizer == "bpe"
        else flow.tokenizer
    )
    return "_".join([
        _run_stamp(),
        "finetune",
        tokenizer_text,
        dataset_name,
        f"lr{_lr_text(train.lr)}",
        f"bn{flow.max_burst_num}",
        f"tl{flow.max_burst_len}",
        f"h{model.hidden_size}",
        f"bs{train.batch_size}",
        f"ep{train.epochs}",
    ])


def _run_file_prefix(run_name):
    return _safe_text(run_name)


def _write_json(path, data):
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_parameter_guide(run_root, file_prefix):
    path = os.path.join(run_root, f"{file_prefix}_parameter_guide.json")
    _write_json(path, PARAMETER_DESCRIPTIONS)
    return path


def _normalize_tokenizer_type(tokenizer_type):
    tokenizer_type = (tokenizer_type or "hex").lower()
    if tokenizer_type not in {"hex", "bpe"}:
        raise ValueError(f"未知 tokenizer: {tokenizer_type}")
    return tokenizer_type


def _normalize_bpe_tokenizer_model(tokenizer_model):
    tokenizer_model = (tokenizer_model or "etbert_wordpiece").lower()

    if tokenizer_model in {"etbert", "wordpiece", "etbert_wordpiece"}:
        return "etbert_wordpiece"

    if tokenizer_model in {
        "etbert_train",
        "wordpiece_train",
        "etbert_wordpiece_train",
        "etbert_wordpiece_trainer",
    }:
        return "etbert_wordpiece_trainer"

    if tokenizer_model in {"bpe", "byte_bpe"}:
        return "bpe"

    raise ValueError(f"未知 bpe_tokenizer_model: {tokenizer_model}")


def _infer_tokenizer_json_model_type(tokenizer_path):
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        return None

    try:
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model", {}).get("type", "").lower() or None
    except Exception:
        return None


def _pcap_jsonl_max_burst_len(flow):
    if flow.jsonl_max_burst_len is not None:
        return int(flow.jsonl_max_burst_len)

    if _normalize_tokenizer_type(flow.tokenizer) == "bpe":
        multiplier = max(1, int(flow.bpe_raw_burst_len_multiplier))
        return int(flow.max_burst_len) * multiplier

    return int(flow.max_burst_len)


def _flow_args(pcap_input, output_jsonl, jsonl_parts_dir, flow, merge_parts=False):
    return SimpleNamespace(
        input=pcap_input,
        output_jsonl=output_jsonl,
        token_bytes=flow.token_bytes,
        max_burst_num=flow.max_burst_num,
        max_burst_len=_pcap_jsonl_max_burst_len(flow),
        min_burst_len=flow.min_burst_len,
        min_burst_num=flow.min_burst_num,
        max_packet_tokens=flow.max_packet_tokens,
        max_packets_per_flow=flow.max_packets_per_flow,
        tcp_idle_timeout=flow.tcp_idle_timeout,
        tcp_handshake_timeout=flow.tcp_handshake_timeout,
        tcp_close_timeout=flow.tcp_close_timeout,
        udp_idle_timeout=flow.udp_idle_timeout,
        cut_link_layer=flow.cut_link_layer,
        label_mode=flow.label_mode,
        label_csv=flow.label_csv,
        csv_encoding=flow.csv_encoding,
        csv_src_ip_col=flow.csv_src_ip_col,
        csv_src_port_col=flow.csv_src_port_col,
        csv_dst_ip_col=flow.csv_dst_ip_col,
        csv_dst_port_col=flow.csv_dst_port_col,
        csv_proto_col=flow.csv_proto_col,
        csv_label_col=flow.csv_label_col,
        drop_unmatched_csv=flow.drop_unmatched_csv,
        num_workers=flow.pcap_num_workers,
        pcap_progress_every=flow.pcap_progress_every,
        pcap_chunksize=flow.pcap_chunksize,
        pcap_maxtasksperchild=flow.pcap_maxtasksperchild,
        shuffle_pcap_files=flow.shuffle_data,
        shuffle_seed=flow.shuffle_seed,
        temp_dir=jsonl_parts_dir,
        merge_parts=merge_parts,
        keep_parts=True,
    )


def _jsonl_to_pt(data_module, input_path, vocab_path, output_dir, flow, file_prefix=""):
    _ensure_dir(output_dir)
    name_prefix = f"{file_prefix}_" if file_prefix else ""
    tokenizer_type = _normalize_tokenizer_type(flow.tokenizer)
    bpe_tokenizer_path = flow.bpe_tokenizer_path if tokenizer_type == "bpe" else None
    bpe_tokenizer_model = (
        _normalize_bpe_tokenizer_model(flow.bpe_tokenizer_model)
        if tokenizer_type == "bpe"
        else None
    )

    jsonl_files = list(data_module.iter_jsonl_files(input_path))
    if not jsonl_files:
        raise FileNotFoundError(f"没有找到 jsonl 文件: {input_path}")
    if flow.shuffle_data:
        rng = random.Random(flow.shuffle_seed)
        rng.shuffle(jsonl_files)
        print(f"[INFO] shuffled jsonl files with seed={flow.shuffle_seed}")

    vocab, special_ids = data_module.load_vocab(
        vocab_path=vocab_path,
        lowercase=not flow.no_lowercase,
    )

    label_map_path = os.path.join(output_dir, f"{name_prefix}label_map.json")
    label_map = data_module.build_or_load_label_map(
        input_path=input_path,
        label_map_path=label_map_path if os.path.exists(label_map_path) else None,
        lowercase_label=flow.lowercase_label,
    )

    args = SimpleNamespace(
        input=input_path,
        vocab_path=vocab_path,
        tokenizer_type=tokenizer_type,
        bpe_tokenizer_path=bpe_tokenizer_path,
        bpe_tokenizer_model=bpe_tokenizer_model,
        jsonl_max_burst_len=_pcap_jsonl_max_burst_len(flow),
        output_dir=output_dir,
        output_prefix=f"{name_prefix}hierarchical",
        manifest_path=os.path.join(output_dir, f"{name_prefix}pt_manifest.jsonl"),
        label_map_path=label_map_path,
        dataset_info_path=os.path.join(output_dir, f"{name_prefix}dataset_info.json"),
        max_burst_num=flow.max_burst_num,
        max_burst_len=flow.max_burst_len,
        shard_size=flow.shard_size,
        num_workers=flow.jsonl_to_pt_workers,
        no_lowercase=flow.no_lowercase,
        lowercase_label=flow.lowercase_label,
        no_cls_sep=flow.no_cls_sep,
        keep_meta=flow.keep_meta,
    )

    args_dict = vars(args).copy()
    tasks = [
        {"jsonl_index": idx, "jsonl_path": path, "args_dict": args_dict}
        for idx, path in enumerate(jsonl_files, start=1)
    ]

    print("\n========== JSONL -> PT ==========")
    print(f"jsonl files: {len(tasks)}")
    print(f"output_dir: {output_dir}")
    print(f"progress_every: {flow.jsonl_progress_every}")

    results = []
    if args.num_workers <= 1:
        data_module.init_worker(
            vocab_path=args.vocab_path,
            label_map=label_map,
            lowercase=not args.no_lowercase,
            tokenizer_type=args.tokenizer_type,
            bpe_tokenizer_path=args.bpe_tokenizer_path,
        )
        for task in tasks:
            result = data_module.process_jsonl_file(task)
            results.append(result)
            if result["error"] is not None:
                print(f"[JSONL {task['jsonl_index']}] ERROR: {result['error']}")
            elif flow.jsonl_progress_every > 0 and (
                len(results) % flow.jsonl_progress_every == 0
                or len(results) == len(tasks)
            ):
                print(
                    f"[JSONL PROGRESS {len(results)}/{len(tasks)}] "
                    f"valid={result['valid_samples']} "
                    f"skipped={result['skipped_samples']} "
                    f"shards={len(result['shards'])}"
                )
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.num_workers,
            initializer=data_module.init_worker,
            initargs=(
                args.vocab_path,
                label_map,
                not args.no_lowercase,
                args.tokenizer_type,
                args.bpe_tokenizer_path,
            ),
        ) as pool:
            for result in pool.imap_unordered(data_module.process_jsonl_file, tasks):
                results.append(result)
                if result["error"] is not None:
                    print(f"[JSONL {result['jsonl_index']}] ERROR: {result['error']}")
                elif flow.jsonl_progress_every > 0 and (
                    len(results) % flow.jsonl_progress_every == 0
                    or len(results) == len(tasks)
                ):
                    print(
                        f"[JSONL PROGRESS {len(results)}/{len(tasks)}] "
                        f"valid={result['valid_samples']} "
                        f"skipped={result['skipped_samples']} "
                        f"shards={len(result['shards'])}"
                    )

    data_module.write_manifest_and_info(
        results=results,
        args=args,
        label_map=label_map,
        special_ids=special_ids,
        vocab_size=len(vocab),
    )

    return {
        "manifest_path": args.manifest_path,
        "dataset_info_path": args.dataset_info_path,
        "label_map_path": args.label_map_path,
    }


def _build_bpe_assets(input_path, tokenizer_dir, flow):
    from process_pretrain_data.build_bpe_vocab_from_jsonl import (
        build_bpe_training_corpus,
        train_bpe_tokenizer,
    )

    _ensure_dir(tokenizer_dir)
    corpus_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            suffix=".txt",
        ) as temp:
            corpus_path = temp.name

        stats = build_bpe_training_corpus(
            input_path=input_path,
            corpus_path=corpus_path,
            lowercase=not flow.no_lowercase,
            unit=flow.bpe_unit,
            tokenizer_model=flow.bpe_tokenizer_model,
        )

        tokenizer_json_path, vocab_txt_path = train_bpe_tokenizer(
            corpus_path=corpus_path,
            output_dir=tokenizer_dir,
            vocab_size=flow.bpe_vocab_size,
            min_frequency=flow.bpe_min_frequency,
            tokenizer_model=flow.bpe_tokenizer_model,
        )

        stats_path = os.path.join(tokenizer_dir, "bpe_training_stats.json")
        _write_json(
            stats_path,
            {
                "tokenizer_type": "bpe",
                "vocab_size": flow.bpe_vocab_size,
                "min_frequency": flow.bpe_min_frequency,
                "tokenizer_model": flow.bpe_tokenizer_model,
                "unit": flow.bpe_unit,
                "lowercase": not flow.no_lowercase,
                "stats": stats,
                "tokenizer_json_path": tokenizer_json_path,
                "vocab_path": vocab_txt_path,
            },
        )

        return {
            "vocab_path": vocab_txt_path,
            "tokenizer_json_path": tokenizer_json_path,
            "stats_path": stats_path,
            "stats": stats,
        }
    finally:
        if corpus_path is not None and os.path.exists(corpus_path):
            os.remove(corpus_path)


def _split_finetune_pt(input_dir, output_dir, options):
    ratio_sum = options.train_ratio + options.valid_ratio + options.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/valid/test 比例之和必须为 1，当前为 {ratio_sum}")

    _ensure_dir(output_dir)
    pt_files = find_pt_files(input_dir)
    if not pt_files:
        raise ValueError(f"没有在目录中找到 .pt 文件: {input_dir}")

    all_samples, file_stats = load_all_samples(pt_files)
    train_set, valid_set, test_set, split_stats = stratified_split(
        samples=all_samples,
        train_ratio=options.train_ratio,
        valid_ratio=options.valid_ratio,
        test_ratio=options.test_ratio,
        seed=options.seed,
    )

    prefix = getattr(options, "file_prefix", "")
    name_prefix = f"{prefix}_" if prefix else ""

    paths = {
        "train_path": os.path.join(output_dir, f"{name_prefix}train.pt"),
        "valid_path": os.path.join(output_dir, f"{name_prefix}valid.pt"),
        "test_path": os.path.join(output_dir, f"{name_prefix}test.pt"),
        "split_info_path": os.path.join(output_dir, f"{name_prefix}split_info.json"),
    }

    save_pt(train_set, paths["train_path"])
    save_pt(valid_set, paths["valid_path"])
    save_pt(test_set, paths["test_path"])

    _write_json(
        paths["split_info_path"],
        {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "num_input_pt_files": len(pt_files),
            "total_samples": len(all_samples),
            "train_ratio": options.train_ratio,
            "valid_ratio": options.valid_ratio,
            "test_ratio": options.test_ratio,
            "seed": options.seed,
            "all_label_counter": count_labels(all_samples),
            "train_label_counter": count_labels(train_set),
            "valid_label_counter": count_labels(valid_set),
            "test_label_counter": count_labels(test_set),
            "split_stats_by_label": split_stats,
            "file_stats": file_stats,
            "first_sample_info": inspect_first_sample(all_samples),
        },
    )

    return paths


def _latest_checkpoint(model_dir):
    candidates = []
    for name in os.listdir(model_dir):
        if not name.startswith("checkpoint_step_") or not name.endswith(".pt"):
            continue
        step_text = name[len("checkpoint_step_"):-len(".pt")]
        try:
            step = int(step_text)
        except ValueError:
            continue
        candidates.append((step, os.path.join(model_dir, name)))

    if not candidates:
        raise FileNotFoundError(f"没有找到 checkpoint_step_*.pt: {model_dir}")

    return max(candidates, key=lambda x: x[0])[1]


def _checkpoint_to_bin(checkpoint_path, output_bin):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = remove_module_prefix(extract_model_state_dict(ckpt))
    _ensure_dir(os.path.dirname(os.path.abspath(output_bin)))
    torch.save(state_dict, output_bin)
    return output_bin


def _build_pretrain_args(pt_info, model_dir, model, train):
    return SimpleNamespace(
        data=pt_info["manifest_path"],
        dataset_info=pt_info["dataset_info_path"],
        output_dir=model_dir,
        hidden_size=model.hidden_size,
        burst_layers=model.burst_layers,
        flow_layers=model.flow_layers,
        num_heads=model.num_heads,
        intermediate_size=model.intermediate_size,
        dropout=model.dropout,
        no_direction_embedding=model.no_direction_embedding,
        token_mlm_probability=train.token_mlm_probability,
        burst_mbm_probability=train.burst_mbm_probability,
        token_loss_weight=train.token_loss_weight,
        burst_loss_weight=train.burst_loss_weight,
        relation_task=train.relation_task,
        relation_pairs=train.relation_pairs,
        nsp_positive_ratio=train.nsp_positive_ratio,
        nsp_loss_weight=train.nsp_loss_weight,
        sodf_loss_weight=train.sodf_loss_weight,
        epochs=train.epochs,
        batch_size=train.batch_size,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        lr=train.lr,
        weight_decay=train.weight_decay,
        max_grad_norm=train.max_grad_norm,
        drop_last=train.drop_last,
        num_workers=train.num_workers,
        device=train.device,
        dist_backend=train.dist_backend,
        ddp_timeout_minutes=train.ddp_timeout_minutes,
        find_unused_parameters=train.find_unused_parameters,
        shuffle=train.shuffle,
        verbose_ranks=train.verbose_ranks,
        fp16=train.fp16,
        seed=train.seed,
        log_every=train.log_every,
        save_every=train.save_every,
        max_steps=train.max_steps,
        plot_loss=train.plot_loss,
    )


def _pretrain_cli_args(args):
    cli = [
        "--data", args.data,
        "--dataset_info", args.dataset_info,
        "--output_dir", args.output_dir,
        "--hidden_size", str(args.hidden_size),
        "--burst_layers", str(args.burst_layers),
        "--flow_layers", str(args.flow_layers),
        "--num_heads", str(args.num_heads),
        "--intermediate_size", str(args.intermediate_size),
        "--dropout", str(args.dropout),
        "--token_mlm_probability", str(args.token_mlm_probability),
        "--burst_mbm_probability", str(args.burst_mbm_probability),
        "--token_loss_weight", str(args.token_loss_weight),
        "--burst_loss_weight", str(args.burst_loss_weight),
        "--relation_task", args.relation_task,
        "--relation_pairs", str(args.relation_pairs),
        "--nsp_positive_ratio", str(args.nsp_positive_ratio),
        "--nsp_loss_weight", str(args.nsp_loss_weight),
        "--sodf_loss_weight", str(args.sodf_loss_weight),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--max_grad_norm", str(args.max_grad_norm),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--dist_backend", args.dist_backend,
        "--ddp_timeout_minutes", str(args.ddp_timeout_minutes),
        "--seed", str(args.seed),
        "--log_every", str(args.log_every),
        "--save_every", str(args.save_every),
        "--max_steps", str(args.max_steps),
    ]

    if args.no_direction_embedding:
        cli.append("--no_direction_embedding")
    if args.drop_last:
        cli.append("--drop_last")
    if args.find_unused_parameters:
        cli.append("--find_unused_parameters")
    else:
        cli.append("--no_find_unused_parameters")
    if args.shuffle:
        cli.append("--shuffle")
    if args.verbose_ranks:
        cli.append("--verbose_ranks")
    if args.fp16:
        cli.append("--fp16")
    if args.plot_loss:
        cli.append("--plot_loss")

    return cli


def _run_pretrain_accum(pt_info, model_dir, model, train):
    if train.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps 必须 >= 1")

    env = os.environ.copy()
    if train.cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(train.cuda_devices)

    pretrain_args = _build_pretrain_args(pt_info, model_dir, model, train)

    if train.nproc_per_node and train.nproc_per_node > 1:
        script_path = os.path.join(
            PROJECT_ROOT,
            "pre-train",
            "pretrain_hierarchical_ddp_grad_accum.py",
        )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(train.nproc_per_node),
            "--master_port",
            str(train.master_port),
            script_path,
        ]
        cmd.extend(_pretrain_cli_args(pretrain_args))
        print("[DDP COMMAND]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
        return

    if train.cuda_devices and "," not in str(train.cuda_devices) and train.device == "auto":
        pretrain_args.device = f"cuda:{train.cuda_devices}"

    old_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if train.cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(train.cuda_devices)
        pretrain_runner.train(pretrain_args)
    finally:
        if old_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible_devices


def _cleanup(path, keep_intermediate):
    if keep_intermediate:
        return
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[CLEAN] removed intermediate: {path}")


def run_pretrain_pipeline(
    pcap_input,
    output_root,
    flow=None,
    model=None,
    train=None,
    keep_intermediate=False,
    run_name=None,
):
    """
    一键完成预训练：PCAP -> JSONL parts -> vocab/tokenizer -> PT shards -> pretrain -> bin。

    最终稳定产物：
        hex: output_root/data/*_hex_vocab.txt
        bpe: output_root/data/*_bpe_tokenizer/bpe_vocab.txt + bpe_tokenizer.json
        output_root/data/pt/pt_manifest.jsonl
        output_root/data/pt/dataset_info.json
        output_root/models/checkpoint_step_xxx.pt
        output_root/models/pretrained_model.bin
    """
    flow = flow or FlowOptions(label_mode="unknown")
    model = model or ModelOptions()
    train = train or PretrainOptions()
    dataset_name = _dataset_name(pcap_input)
    run_name = _pretrain_run_name(flow, model, train, dataset_name, run_name=run_name)
    file_prefix = _run_file_prefix(run_name)

    output_root = os.path.abspath(output_root)
    run_root = _ensure_dir(os.path.join(output_root, run_name))
    raw_dir = _ensure_dir(os.path.join(run_root, "raw"))
    data_dir = _ensure_dir(os.path.join(run_root, "data"))
    pt_dir = _ensure_dir(os.path.join(data_dir, "pt"))
    model_dir = _ensure_dir(os.path.join(run_root, "models"))

    jsonl_parts_dir = os.path.join(raw_dir, f"{file_prefix}_jsonl_parts")
    output_jsonl = os.path.join(raw_dir, f"{file_prefix}_flow_burst_unused.jsonl")
    flow.tokenizer = _normalize_tokenizer_type(flow.tokenizer)
    flow.bpe_tokenizer_model = _normalize_bpe_tokenizer_model(flow.bpe_tokenizer_model)
    if flow.tokenizer == "bpe":
        bpe_tokenizer_dir = os.path.join(data_dir, f"{file_prefix}_bpe_tokenizer")
        flow.bpe_tokenizer_path = os.path.join(bpe_tokenizer_dir, "bpe_tokenizer.json")
        vocab_path = os.path.join(bpe_tokenizer_dir, "bpe_vocab.txt")
    else:
        bpe_tokenizer_dir = None
        vocab_path = os.path.join(data_dir, f"{file_prefix}_hex_vocab.txt")
    parameter_guide_path = _write_parameter_guide(run_root, file_prefix)

    _write_json(
        os.path.join(run_root, f"{file_prefix}_pipeline_config.json"),
        {
            "pipeline": "pretrain",
            "run_name": run_name,
            "dataset_name": dataset_name,
            "created_at": _run_stamp(),
            "pcap_input": pcap_input,
            "output_root": output_root,
            "run_root": run_root,
            "flow": asdict(flow),
            "model": asdict(model),
            "train": asdict(train),
            "parameter_guide_path": parameter_guide_path,
            "keep_intermediate": keep_intermediate,
        },
    )

    print("\n========== PRETRAIN: PCAP -> JSONL PARTS ==========")
    write_pretrain_jsonl(
        _flow_args(
            pcap_input=pcap_input,
            output_jsonl=output_jsonl,
            jsonl_parts_dir=jsonl_parts_dir,
            flow=flow,
            merge_parts=False,
        )
    )

    if flow.tokenizer == "bpe":
        print("\n========== PRETRAIN: BUILD BPE TOKENIZER ==========")
        bpe_assets = _build_bpe_assets(
            input_path=jsonl_parts_dir,
            tokenizer_dir=bpe_tokenizer_dir,
            flow=flow,
        )
        vocab_path = bpe_assets["vocab_path"]
        flow.bpe_tokenizer_path = bpe_assets["tokenizer_json_path"]
    else:
        bpe_assets = None
        print("\n========== PRETRAIN: BUILD HEX VOCAB ==========")
        build_vocab(
            input_path=jsonl_parts_dir,
            output_vocab=vocab_path,
            min_freq=flow.min_freq,
            max_vocab_size=flow.max_vocab_size,
            lowercase=not flow.no_lowercase,
        )

    pt_info = _jsonl_to_pt(
        data_module=pretrain_data,
        input_path=jsonl_parts_dir,
        vocab_path=vocab_path,
        output_dir=pt_dir,
        flow=flow,
        file_prefix=file_prefix,
    )

    _cleanup(jsonl_parts_dir, keep_intermediate)

    print("\n========== PRETRAIN: TRAIN ==========")
    _run_pretrain_accum(
        pt_info=pt_info,
        model_dir=model_dir,
        model=model,
        train=train,
    )

    checkpoint_path = _latest_checkpoint(model_dir)
    pretrained_bin = _checkpoint_to_bin(
        checkpoint_path=checkpoint_path,
        output_bin=os.path.join(model_dir, f"{file_prefix}_pretrained_model.bin"),
    )

    result = {
        "run_name": run_name,
        "dataset_name": dataset_name,
        "run_root": run_root,
        "vocab_path": vocab_path,
        "tokenizer_type": flow.tokenizer,
        "bpe_tokenizer_path": flow.bpe_tokenizer_path,
        "bpe_tokenizer_model": flow.bpe_tokenizer_model,
        "bpe_stats_path": bpe_assets["stats_path"] if bpe_assets else None,
        "jsonl_max_burst_len": _pcap_jsonl_max_burst_len(flow),
        "manifest_path": pt_info["manifest_path"],
        "dataset_info_path": pt_info["dataset_info_path"],
        "label_map_path": pt_info["label_map_path"],
        "checkpoint_path": checkpoint_path,
        "pretrained_model_path": pretrained_bin,
        "model_dir": model_dir,
        "parameter_guide_path": parameter_guide_path,
        "flow": asdict(flow),
        "model": asdict(model),
        "train": asdict(train),
    }
    _write_json(os.path.join(run_root, f"{file_prefix}_pipeline_outputs.json"), result)
    return result


def run_finetune_pipeline(
    pcap_input,
    output_root,
    pretrained_model_path,
    vocab_path,
    dataset_info_path,
    flow=None,
    model=None,
    train=None,
    keep_intermediate=False,
    run_name=None,
):
    """
    一键完成微调：PCAP -> JSONL parts -> PT shards -> split -> finetune/test。

    微调阶段复用预训练阶段的 vocab 和 dataset_info，避免重复生成 token/id 规则。
    """
    flow = flow or FlowOptions(label_mode="parent")
    model = model or ModelOptions()
    train = train or FinetuneOptions()

    with open(dataset_info_path, "r", encoding="utf-8") as f:
        base_dataset_info = json.load(f)
    flow.max_burst_num = int(base_dataset_info["max_burst_num"])
    flow.max_burst_len = int(base_dataset_info["max_burst_len"])
    flow.tokenizer = _normalize_tokenizer_type(base_dataset_info.get("tokenizer_type", "hex"))
    base_bpe_tokenizer_model = base_dataset_info.get("bpe_tokenizer_model")
    if not base_bpe_tokenizer_model:
        base_bpe_model_type = str(base_dataset_info.get("bpe_tokenizer_model_type", "")).lower()
        if base_bpe_model_type == "wordpiece":
            base_bpe_tokenizer_model = "etbert_wordpiece"
        elif base_bpe_model_type == "bpe":
            base_bpe_tokenizer_model = "bpe"
    flow.bpe_tokenizer_model = _normalize_bpe_tokenizer_model(
        base_bpe_tokenizer_model or flow.bpe_tokenizer_model
    )
    flow.jsonl_max_burst_len = base_dataset_info.get("jsonl_max_burst_len", flow.jsonl_max_burst_len)
    flow.bpe_tokenizer_path = base_dataset_info.get("bpe_tokenizer_path", flow.bpe_tokenizer_path)

    base_vocab_path = base_dataset_info.get("vocab_path")
    if base_vocab_path and os.path.exists(base_vocab_path):
        if os.path.abspath(vocab_path) != os.path.abspath(base_vocab_path):
            print(f"[WARN] vocab_path 与 dataset_info 不一致，使用 dataset_info 中的词表: {base_vocab_path}")
        vocab_path = base_vocab_path

    if flow.tokenizer == "bpe":
        bpe_tokenizer_candidates = [
            flow.bpe_tokenizer_path,
            os.path.join(os.path.dirname(os.path.abspath(vocab_path)), "bpe_tokenizer.json"),
        ]
        flow.bpe_tokenizer_path = next(
            (path for path in bpe_tokenizer_candidates if path and os.path.exists(path)),
            flow.bpe_tokenizer_path,
        )

        if not flow.bpe_tokenizer_path or not os.path.exists(flow.bpe_tokenizer_path):
            raise ValueError("预训练 dataset_info.json 标记为 BPE，但找不到 bpe_tokenizer.json")

        if not base_dataset_info.get("bpe_tokenizer_model"):
            inferred_model_type = _infer_tokenizer_json_model_type(flow.bpe_tokenizer_path)
            if inferred_model_type == "wordpiece":
                flow.bpe_tokenizer_model = "etbert_wordpiece"
            elif inferred_model_type == "bpe":
                flow.bpe_tokenizer_model = "bpe"

    dataset_name = _dataset_name(pcap_input)
    run_name = _finetune_run_name(flow, model, train, dataset_name, run_name=run_name)
    file_prefix = _run_file_prefix(run_name)
    train.file_prefix = file_prefix

    output_root = os.path.abspath(output_root)
    run_root = _ensure_dir(os.path.join(output_root, run_name))
    raw_dir = _ensure_dir(os.path.join(run_root, "raw"))
    data_dir = _ensure_dir(os.path.join(run_root, "data"))
    pt_dir = _ensure_dir(os.path.join(data_dir, "pt"))
    split_dir = _ensure_dir(os.path.join(data_dir, "split"))
    model_dir = _ensure_dir(os.path.join(run_root, "models"))

    jsonl_parts_dir = os.path.join(raw_dir, f"{file_prefix}_jsonl_parts")
    output_jsonl = os.path.join(raw_dir, f"{file_prefix}_flow_burst_unused.jsonl")
    parameter_guide_path = _write_parameter_guide(run_root, file_prefix)

    _write_json(
        os.path.join(run_root, f"{file_prefix}_pipeline_config.json"),
        {
            "pipeline": "finetune",
            "run_name": run_name,
            "dataset_name": dataset_name,
            "created_at": _run_stamp(),
            "pcap_input": pcap_input,
            "output_root": output_root,
            "run_root": run_root,
            "pretrained_model_path": pretrained_model_path,
            "vocab_path": vocab_path,
            "dataset_info_path": dataset_info_path,
            "flow": asdict(flow),
            "model": asdict(model),
            "train": asdict(train),
            "parameter_guide_path": parameter_guide_path,
            "keep_intermediate": keep_intermediate,
        },
    )

    print("\n========== FINETUNE: PCAP -> JSONL PARTS ==========")
    write_finetune_jsonl(
        _flow_args(
            pcap_input=pcap_input,
            output_jsonl=output_jsonl,
            jsonl_parts_dir=jsonl_parts_dir,
            flow=flow,
            merge_parts=False,
        )
    )

    pt_info = _jsonl_to_pt(
        data_module=finetune_data,
        input_path=jsonl_parts_dir,
        vocab_path=vocab_path,
        output_dir=pt_dir,
        flow=flow,
        file_prefix=file_prefix,
    )

    split_paths = _split_finetune_pt(
        input_dir=pt_dir,
        output_dir=split_dir,
        options=train,
    )

    _cleanup(jsonl_parts_dir, keep_intermediate)
    _cleanup(pt_dir, keep_intermediate)

    print("\n========== FINETUNE: TRAIN / VALID / TEST ==========")
    finetune_runner.train(
        SimpleNamespace(
            train_path=split_paths["train_path"],
            valid_path=split_paths["valid_path"],
            test_path=split_paths["test_path"],
            dataset_info=dataset_info_path,
            pretrained_model_path=pretrained_model_path,
            output_dir=model_dir,
            hidden_size=model.hidden_size,
            burst_layers=model.burst_layers,
            flow_layers=model.flow_layers,
            num_heads=model.num_heads,
            intermediate_size=model.intermediate_size,
            dropout=model.dropout,
            no_direction_embedding=model.no_direction_embedding,
            epochs=train.epochs,
            batch_size=train.batch_size,
            lr=train.lr,
            weight_decay=train.weight_decay,
            max_grad_norm=train.max_grad_norm,
            drop_last=train.drop_last,
            max_steps=train.max_steps,
            num_workers=train.num_workers,
            device=train.device,
            fp16=train.fp16,
            seed=train.seed,
            log_every=train.log_every,
        )
    )

    fixed_model_path = os.path.join(model_dir, "finetuned_model.bin")
    prefixed_model_path = os.path.join(model_dir, f"{file_prefix}_finetuned_model.bin")
    if os.path.exists(fixed_model_path):
        os.replace(fixed_model_path, prefixed_model_path)

    result = {
        "run_name": run_name,
        "dataset_name": dataset_name,
        "run_root": run_root,
        "train_path": split_paths["train_path"],
        "valid_path": split_paths["valid_path"],
        "test_path": split_paths["test_path"],
        "split_info_path": split_paths["split_info_path"],
        "finetuned_model_path": prefixed_model_path,
        "model_dir": model_dir,
        "pt_dataset_info_path": pt_info["dataset_info_path"] if keep_intermediate else None,
        "pt_label_map_path": pt_info["label_map_path"] if keep_intermediate else None,
        "pretrained_model_path": pretrained_model_path,
        "vocab_path": vocab_path,
        "tokenizer_type": flow.tokenizer,
        "bpe_tokenizer_path": flow.bpe_tokenizer_path,
        "bpe_tokenizer_model": flow.bpe_tokenizer_model,
        "jsonl_max_burst_len": _pcap_jsonl_max_burst_len(flow),
        "dataset_info_path": dataset_info_path,
        "parameter_guide_path": parameter_guide_path,
        "flow": asdict(flow),
        "model": asdict(model),
        "train": asdict(train),
    }
    _write_json(os.path.join(run_root, f"{file_prefix}_pipeline_outputs.json"), result)
    return result


def _add_flow_args(parser, default_label_mode):
    parser.add_argument("--pcap_input", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--run_name",
        default=None,
        help="可选实验名；不传时自动使用 时间+学习率+burst数量+token数量 等参数生成。",
    )
    parser.add_argument("--tokenizer", choices=["hex", "bpe"], default="hex")
    parser.add_argument("--token_bytes", type=int, default=2)
    parser.add_argument(
        "--jsonl_max_burst_len",
        type=int,
        default=None,
        help="PCAP->JSONL 阶段每个 burst 保留多少固定 hex token。默认 hex=max_burst_len，BPE=max_burst_len*bpe_raw_burst_len_multiplier。",
    )
    parser.add_argument("--max_burst_num", type=int, default=16)
    parser.add_argument("--max_burst_len", type=int, default=128)
    parser.add_argument("--min_burst_len", type=int, default=1)
    parser.add_argument("--min_burst_num", type=int, default=2)
    parser.add_argument("--max_packet_tokens", type=int, default=None)
    parser.add_argument("--max_packets_per_flow", type=int, default=None)
    parser.add_argument(
        "--tcp_idle_timeout",
        type=float,
        default=120.0,
        help="TCP 空闲多久后切为新会话；<=0 表示关闭该超时。",
    )
    parser.add_argument(
        "--tcp_handshake_timeout",
        type=float,
        default=30.0,
        help="TCP 半连接超时秒数；<=0 表示关闭。",
    )
    parser.add_argument(
        "--tcp_close_timeout",
        type=float,
        default=15.0,
        help="TCP 出现 FIN 后等待完整关闭的秒数；<=0 表示关闭。",
    )
    parser.add_argument(
        "--udp_idle_timeout",
        type=float,
        default=30.0,
        help="UDP 空闲多久后切为新会话；<=0 表示关闭。",
    )
    parser.add_argument("--no_cut_link_layer", action="store_true")
    parser.add_argument("--label_mode", choices=["unknown", "parent", "csv"], default=default_label_mode)
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--drop_unmatched_csv", action="store_true")
    parser.add_argument("--pcap_num_workers", type=int, default=48)
    parser.add_argument(
        "--pcap_progress_every",
        type=int,
        default=100,
        help="PCAP 转 JSONL 时每处理多少个 pcap 打印一次进度；0 表示只打印错误和汇总。",
    )
    parser.add_argument(
        "--pcap_chunksize",
        type=int,
        default=32,
        help="PCAP 多进程任务批量分发大小。小 pcap 很多时建议 16/32/64。",
    )
    parser.add_argument(
        "--pcap_maxtasksperchild",
        type=int,
        default=256,
        help="每个 PCAP worker 处理多少个任务后重启。默认 None 不重启；不要为小 pcap 设置 1。",
    )
    parser.add_argument(
        "--no_shuffle_data",
        action="store_true",
        help="关闭数据准备阶段的文件顺序打乱。默认会打乱 PCAP 和 JSONL 文件顺序。",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="数据准备阶段打乱顺序使用的随机种子。",
    )
    parser.add_argument("--jsonl_to_pt_workers", type=int, default=48)
    parser.add_argument(
        "--jsonl_progress_every",
        type=int,
        default=100,
        help="JSONL 转 PT 时每处理多少个 JSONL 打印一次进度；0 表示只打印错误和汇总。",
    )
    parser.add_argument("--shard_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_vocab_size", type=int, default=None)
    parser.add_argument("--bpe_vocab_size", type=int, default=65541)
    parser.add_argument("--bpe_min_frequency", type=int, default=2)
    parser.add_argument(
        "--bpe_tokenizer_model",
        choices=["etbert_wordpiece", "etbert_wordpiece_trainer", "bpe"],
        default="etbert_wordpiece",
        help="tokenizer=bpe 时的实际算法：默认 etbert_wordpiece 固定 0000-ffff；etbert_wordpiece_trainer 调用 WordPieceTrainer；bpe 表示旧的连续 hex BPE。",
    )
    parser.add_argument("--bpe_unit", choices=["burst", "flow"], default="burst")
    parser.add_argument("--bpe_raw_burst_len_multiplier", type=int, default=2)
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--lowercase_label", action="store_true")
    parser.add_argument("--no_cls_sep", action="store_true")
    parser.add_argument("--keep_meta", action="store_true")
    parser.add_argument("--keep_intermediate", action="store_true")


def _add_model_args(parser):
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--burst_layers", type=int, default=8)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_direction_embedding", action="store_true")


def _flow_from_args(args):
    return FlowOptions(
        tokenizer=args.tokenizer,
        token_bytes=args.token_bytes,
        jsonl_max_burst_len=args.jsonl_max_burst_len,
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
        cut_link_layer=not args.no_cut_link_layer,
        label_mode=args.label_mode,
        label_csv=args.label_csv,
        drop_unmatched_csv=args.drop_unmatched_csv,
        pcap_num_workers=args.pcap_num_workers,
        pcap_progress_every=args.pcap_progress_every,
        pcap_chunksize=args.pcap_chunksize,
        pcap_maxtasksperchild=args.pcap_maxtasksperchild,
        shuffle_data=not args.no_shuffle_data,
        shuffle_seed=args.shuffle_seed,
        jsonl_to_pt_workers=args.jsonl_to_pt_workers,
        jsonl_progress_every=args.jsonl_progress_every,
        shard_size=args.shard_size,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_min_frequency=args.bpe_min_frequency,
        bpe_tokenizer_model=args.bpe_tokenizer_model,
        bpe_unit=args.bpe_unit,
        bpe_raw_burst_len_multiplier=args.bpe_raw_burst_len_multiplier,
        no_lowercase=args.no_lowercase,
        lowercase_label=args.lowercase_label,
        no_cls_sep=args.no_cls_sep,
        keep_meta=args.keep_meta,
    )


def _model_from_args(args):
    return ModelOptions(
        hidden_size=args.hidden_size,
        burst_layers=args.burst_layers,
        flow_layers=args.flow_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        no_direction_embedding=args.no_direction_embedding,
    )


def main():
    parser = argparse.ArgumentParser(
        description="统一入口：预训练流水线或微调流水线。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain_parser = subparsers.add_parser("pretrain")
    _add_flow_args(pretrain_parser, default_label_mode="unknown")
    _add_model_args(pretrain_parser)
    pretrain_parser.add_argument("--epochs", type=int, default=100)
    pretrain_parser.add_argument("--batch_size", type=int, default=16)
    pretrain_parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数；例如 4 表示每 4 个 micro batch 更新一次参数。",
    )
    pretrain_parser.add_argument("--lr", type=float, default=2e-5)
    pretrain_parser.add_argument("--max_steps", type=int, default=100000)
    pretrain_parser.add_argument("--save_every", type=int, default=5000)
    pretrain_parser.add_argument("--log_every", type=int, default=10)
    pretrain_parser.add_argument("--seed", type=int, default=42)
    pretrain_parser.add_argument("--relation_task", choices=["none", "nsp", "sodf", "both"], default="sodf")
    pretrain_parser.add_argument("--relation_pairs", type=int, default=32)
    pretrain_parser.add_argument("--device", default="auto")
    pretrain_parser.add_argument(
        "--cuda_devices",
        default=None,
        help="指定可见 CUDA，例如 0 或 0,1,2,3；会写入 CUDA_VISIBLE_DEVICES。",
    )
    pretrain_parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="DDP 进程数。单卡/单进程为 1；多卡通常等于 cuda 数量。",
    )
    pretrain_parser.add_argument("--master_port", type=int, default=29500)
    pretrain_parser.add_argument("--dist_backend", choices=["auto", "nccl", "gloo"], default="auto")
    pretrain_parser.add_argument("--ddp_timeout_minutes", type=int, default=60)
    pretrain_parser.add_argument("--no_find_unused_parameters", action="store_true")
    pretrain_parser.add_argument("--shuffle", action="store_true")
    pretrain_parser.add_argument(
        "--verbose_ranks",
        action="store_true",
        help="打印所有 DDP rank 的调试信息；默认只打印 rank0。",
    )
    pretrain_parser.add_argument("--no_fp16", action="store_true")

    finetune_parser = subparsers.add_parser("finetune")
    _add_flow_args(finetune_parser, default_label_mode="parent")
    _add_model_args(finetune_parser)
    finetune_parser.add_argument("--pretrained_model_path", required=True)
    finetune_parser.add_argument("--vocab_path", required=True)
    finetune_parser.add_argument("--dataset_info_path", required=True)
    finetune_parser.add_argument("--epochs", type=int, default=10)
    finetune_parser.add_argument("--batch_size", type=int, default=16)
    finetune_parser.add_argument("--lr", type=float, default=1e-4)
    finetune_parser.add_argument("--max_steps", type=int, default=-1)
    finetune_parser.add_argument("--log_every", type=int, default=10)
    finetune_parser.add_argument("--seed", type=int, default=42)
    finetune_parser.add_argument("--train_ratio", type=float, default=0.8)
    finetune_parser.add_argument("--valid_ratio", type=float, default=0.1)
    finetune_parser.add_argument("--test_ratio", type=float, default=0.1)
    finetune_parser.add_argument("--device", default="auto")
    finetune_parser.add_argument("--no_fp16", action="store_true")

    args = parser.parse_args()
    flow = _flow_from_args(args)
    model = _model_from_args(args)

    if args.command == "pretrain":
        result = run_pretrain_pipeline(
            pcap_input=args.pcap_input,
            output_root=args.output_root,
            flow=flow,
            model=model,
            train=PretrainOptions(
                epochs=args.epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                lr=args.lr,
                max_steps=args.max_steps,
                save_every=args.save_every,
                log_every=args.log_every,
                seed=args.seed,
                relation_task=args.relation_task,
                relation_pairs=args.relation_pairs,
                device=args.device,
                cuda_devices=args.cuda_devices,
                nproc_per_node=args.nproc_per_node,
                master_port=args.master_port,
                dist_backend=args.dist_backend,
                ddp_timeout_minutes=args.ddp_timeout_minutes,
                find_unused_parameters=not args.no_find_unused_parameters,
                shuffle=args.shuffle,
                verbose_ranks=args.verbose_ranks,
                fp16=not args.no_fp16,
            ),
            keep_intermediate=args.keep_intermediate,
            run_name=args.run_name,
        )
    else:
        result = run_finetune_pipeline(
            pcap_input=args.pcap_input,
            output_root=args.output_root,
            pretrained_model_path=args.pretrained_model_path,
            vocab_path=args.vocab_path,
            dataset_info_path=args.dataset_info_path,
            flow=flow,
            model=model,
            train=FinetuneOptions(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_steps=args.max_steps,
                log_every=args.log_every,
                seed=args.seed,
                train_ratio=args.train_ratio,
                valid_ratio=args.valid_ratio,
                test_ratio=args.test_ratio,
                device=args.device,
                fp16=not args.no_fp16,
            ),
            keep_intermediate=args.keep_intermediate,
            run_name=args.run_name,
        )

    print("\n========== PIPELINE OUTPUTS ==========")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
