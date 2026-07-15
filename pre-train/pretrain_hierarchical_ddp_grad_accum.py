import os
import sys
import json
import time
import argparse
import csv
from datetime import timedelta
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# 当前文件在 pre-train 目录下，所以项目根目录就是上一级。
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 让所有相对路径都从项目根目录开始解析。
os.chdir(PROJECT_ROOT)

if os.environ.get("HF_VERBOSE_IMPORT", "0") == "1" and int(os.environ.get("RANK", "0")) == 0:
    print("[SCRIPT_DIR]", SCRIPT_DIR, flush=True)
    print("[PROJECT_ROOT]", PROJECT_ROOT, flush=True)
    print("[REAL_CWD]", os.getcwd(), flush=True)

from process_pretrain_data.hierarchical_dataset import (
    load_dataset,
    load_dataset_info,
    HierarchicalCollator,
)

from uer.models.hierarchical_burst_flow_bert_uer import HierarchicalBurstFlowBertUER
from uer.utils.relation_sampler import sample_nsp_pairs, sample_sodf_pairs


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def is_main_process():
    return get_rank() == 0


def setup_distributed(args):
    """
    支持两种启动方式：
    1. 单卡/CPU：python pretrain_hierarchical_ddp.py
    2. 多卡 DDP：torchrun --nproc_per_node=2 pretrain_hierarchical_ddp.py

    注意：
    - torchrun 会自动提供 RANK / WORLD_SIZE / LOCAL_RANK。
    - 多卡时每个 rank 都必须执行：初始化分布式、构建模型、DDP 包装、optimizer 创建、DataLoader 创建。
    - 只有打印、保存 checkpoint、画 loss 图只放在 rank0。
    """
    args.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if args.distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
        else:
            device = torch.device("cpu")

        if args.dist_backend == "auto":
            backend = "nccl" if (torch.cuda.is_available() and dist.is_nccl_available() and os.name != "nt") else "gloo"
        else:
            backend = args.dist_backend

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=args.ddp_timeout_minutes),
        )
        args.dist_backend_used = backend
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.dist_backend_used = "none"

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

    return device


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        # 这里不要再 barrier，避免某个 rank 已经异常退出时其它 rank 卡死。
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, step, args, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model

    ckpt = {
        "step": step,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }

    path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
    torch.save(ckpt, path)

    print(f"[SAVE] {path}")


def build_model(args, dataset_info):
    special_ids = dataset_info["special_ids"]

    model = HierarchicalBurstFlowBertUER(
        vocab_size=int(dataset_info["vocab_size"]),
        max_burst_num=int(dataset_info["max_burst_num"]),
        max_burst_len=int(dataset_info["max_burst_len"]),
        hidden_size=args.hidden_size,
        burst_layers=args.burst_layers,
        flow_layers=args.flow_layers,
        heads_num=args.num_heads,
        feedforward_size=args.intermediate_size,
        dropout=args.dropout,
        attention_dropout=args.dropout,
        pad_id=int(special_ids["pad_id"]),
        mask_id=int(special_ids["mask_id"]),
        direction_vocab_size=3,
        use_direction_embedding=not args.no_direction_embedding,
        sodf_classes=5,
    )

    return model


def move_batch_to_device(batch, device):
    new_batch = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v

    return new_batch


def resolve_arg_path(path):
    """把命令行传入的相对路径统一解析到项目根目录下面。"""
    if path is None:
        return None

    if os.path.isabs(path):
        return path

    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def build_relation_tasks(args, batch):
    """
    根据 relation_task 动态构造 NSP / SODF 关系任务。

    batch 已经被 move 到 device 上，所以采样得到的 pairs/labels
    也会在同一个 device 上。
    """

    nsp_pairs = None
    nsp_labels = None
    sodf_pairs = None
    sodf_labels = None

    if args.relation_task in ["nsp", "both"]:
        nsp_pairs, nsp_labels = sample_nsp_pairs(
            burst_mask=batch["burst_mask"],
            num_pairs=args.relation_pairs,
            positive_ratio=args.nsp_positive_ratio,
        )

    if args.relation_task in ["sodf", "both"]:
        sodf_pairs, sodf_labels = sample_sodf_pairs(
            burst_mask=batch["burst_mask"],
            direction=batch["direction"],
            num_pairs=args.relation_pairs,
        )

    return nsp_pairs, nsp_labels, sodf_pairs, sodf_labels

class LossPlotter:
    """
    训练过程中记录并实时保存 5 张 loss 曲线图。

    每次 update() 会：
    1. 追加一行 loss_log.csv
    2. 更新 5 张单独的 loss 曲线图
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.csv_path = os.path.join(output_dir, "loss_log.csv")

        self.png_paths = {
            "loss": os.path.join(output_dir, "loss_curve_total.png"),
            "token_mbm": os.path.join(output_dir, "loss_curve_token_mbm.png"),
            "burst_mbm": os.path.join(output_dir, "loss_curve_burst_mbm.png"),
            "nsp": os.path.join(output_dir, "loss_curve_nsp.png"),
            "sodf": os.path.join(output_dir, "loss_curve_sodf.png"),
        }

        self.steps = []
        self.loss = []
        self.token_mbm = []
        self.burst_mbm = []
        self.nsp = []
        self.sodf = []

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "loss",
                "token_mbm",
                "burst_mbm",
                "nsp",
                "sodf",
            ])

    def update(
        self,
        step,
        loss,
        token_mbm,
        burst_mbm,
        nsp,
        sodf,
    ):
        self.steps.append(int(step))
        self.loss.append(float(loss))
        self.token_mbm.append(float(token_mbm))
        self.burst_mbm.append(float(burst_mbm))
        self.nsp.append(float(nsp))
        self.sodf.append(float(sodf))

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(step),
                float(loss),
                float(token_mbm),
                float(burst_mbm),
                float(nsp),
                float(sodf),
            ])

        # 每次 update 都更新图片。
        # 因为 update 只会在 global_step % log_every == 0 时调用，
        # 所以实际效果就是每隔 log_every 步更新一次图片。
        self.plot_all()

    def plot_one(self, values, name, ylabel, title, png_path):
        if len(self.steps) == 0:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, values, label=name)

        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        plt.savefig(png_path, dpi=150)
        plt.close()

    def plot_all(self):
        self.plot_one(
            values=self.loss,
            name="loss",
            ylabel="Loss",
            title="Total Loss Curve",
            png_path=self.png_paths["loss"],
        )

        self.plot_one(
            values=self.token_mbm,
            name="token_mbm",
            ylabel="Token MBM Loss",
            title="Token-level MBM Loss Curve",
            png_path=self.png_paths["token_mbm"],
        )

        self.plot_one(
            values=self.burst_mbm,
            name="burst_mbm",
            ylabel="Burst MBM Loss",
            title="Burst-level MBM Loss Curve",
            png_path=self.png_paths["burst_mbm"],
        )

        self.plot_one(
            values=self.nsp,
            name="nsp",
            ylabel="NSP Loss",
            title="NSP Loss Curve",
            png_path=self.png_paths["nsp"],
        )

        self.plot_one(
            values=self.sodf,
            name="sodf",
            ylabel="SODF Loss",
            title="SODF Loss Curve",
            png_path=self.png_paths["sodf"],
        )

def train(args):
    set_seed(args.seed)

    device = setup_distributed(args)

    if is_main_process() or args.verbose_ranks:
        print(
            f"[RANK INIT] rank={args.rank} local_rank={args.local_rank} "
            f"world_size={args.world_size} device={device} backend={args.dist_backend_used}",
            flush=True,
        )

    if is_main_process():
        print("[DEVICE]", device, flush=True)
        print("[DISTRIBUTED]", args.distributed, flush=True)
        print("[WORLD_SIZE]", args.world_size, flush=True)
        print("[RANK]", args.rank, flush=True)
        print("[LOCAL_RANK]", args.local_rank, flush=True)
        print("[DIST_BACKEND]", args.dist_backend_used, flush=True)

    # 先加载 dataset_info。它很小，模型结构依赖它。
    if is_main_process():
        print(f"[LOAD] loading dataset_info: {args.dataset_info}", flush=True)

    dataset_info = load_dataset_info(args.dataset_info)

    if is_main_process():
        print("[LOAD] dataset_info loaded", flush=True)

    # 重点修复：
    # 先在所有 rank 上构建模型并包 DDP，再加载大数据集。
    # 避免某个 rank 加载数据太慢，其他 rank 已经进入 DDP 初始化，最终 NCCL ALLGATHER 超时。
    model = build_model(args, dataset_info)
    model.to(device)

    param_tensor_count = len(list(model.parameters()))
    total_param_count = sum(p.numel() for p in model.parameters())

    if is_main_process() or args.verbose_ranks:
        print(
            f"[MODEL DEBUG BEFORE DDP] rank={args.rank} local_rank={args.local_rank} "
            f"param_tensors={param_tensor_count} total_params={total_param_count}",
            flush=True,
        )

    if param_tensor_count == 0:
        raise RuntimeError(
            f"rank={args.rank} model has 0 parameters before DDP. "
            f"Please check build_model() and HierarchicalBurstFlowBertUER."
        )

    if args.distributed:
        # 确保所有 rank 都已经完成模型构建，再一起进入 DDP。
        #dist.barrier()

        if device.type == "cuda":
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=args.find_unused_parameters,
            )
        else:
            model = DDP(
                model,
                find_unused_parameters=args.find_unused_parameters,
            )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    use_amp = args.fp16 and device.type == "cuda"

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if is_main_process():
        print("[MODEL]", flush=True)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"total params: {total_params:,}", flush=True)
        print(f"trainable params: {trainable_params:,}", flush=True)

    # DDP 已经初始化完成后，再加载大数据集。
    # 这样就算数据加载慢，也不会卡在 DDP 参数校验那里。
    if is_main_process():
        print(f"[LOAD] loading dataset: {args.data}", flush=True)

    dataset = load_dataset(args.data)

    if is_main_process():
        print("[LOAD] dataset loaded", flush=True)

    collator = HierarchicalCollator(
        dataset_info=dataset_info,
        token_mlm_probability=args.token_mlm_probability,
        burst_mbm_probability=args.burst_mbm_probability,
    )

    if args.distributed:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=args.shuffle,
            drop_last=args.drop_last,
        )
    else:
        train_sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None and args.shuffle),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=args.drop_last,
        persistent_workers=(args.num_workers > 0),
    )

    if is_main_process():
        print("[DATA]", flush=True)
        print(f"samples: {len(dataset)}", flush=True)
        print(f"batch_size_per_gpu: {args.batch_size}", flush=True)
        print(f"world_size: {args.world_size}", flush=True)
        print(f"global_batch_size: {args.batch_size * args.world_size}", flush=True)
        print(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}", flush=True)
        print(f"effective_global_batch_size: {args.batch_size * args.world_size * args.gradient_accumulation_steps}", flush=True)
        print(f"steps per epoch per process: {len(dataloader)}", flush=True)
        print(f"optimizer updates per epoch per process: {(len(dataloader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps}", flush=True)

        print("[TASK]", flush=True)
        print(f"relation_task: {args.relation_task}", flush=True)
        print(f"token_mlm_probability: {args.token_mlm_probability}", flush=True)
        print(f"burst_mbm_probability: {args.burst_mbm_probability}", flush=True)
        print(f"relation_pairs: {args.relation_pairs}", flush=True)

    global_step = 0
    start_time = time.time()

    loss_plotter = None
    if args.plot_loss and is_main_process():
        loss_plotter = LossPlotter(output_dir=args.output_dir)
        print("[PLOT]", flush=True)
        print(f"loss csv: {loss_plotter.csv_path}", flush=True)
        for name, path in loss_plotter.png_paths.items():
            print(f"{name} png: {path}", flush=True)

    model.train()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n[EPOCH {epoch}/{args.epochs}]", flush=True)

        for micro_step, batch in enumerate(dataloader):
            batch = move_batch_to_device(batch, device)

            nsp_pairs, nsp_labels, sodf_pairs, sodf_labels = build_relation_tasks(
                args=args,
                batch=batch,
            )

            is_accum_step = (micro_step + 1) % args.gradient_accumulation_steps == 0
            is_last_step = (micro_step + 1) == len(dataloader)
            should_update = is_accum_step or is_last_step

            # DDP 梯度累计时，非更新步不做梯度同步，减少通信开销。
            if args.distributed and hasattr(model, "no_sync") and not should_update:
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()

            with sync_context:
                if use_amp:
                    try:
                        autocast_ctx = torch.amp.autocast("cuda", enabled=True)
                    except Exception:
                        autocast_ctx = torch.cuda.amp.autocast(enabled=True)

                    with autocast_ctx:
                        outputs = model(
                            src=batch["token_mbm_input"],
                            token_mask=batch["token_mask"],
                            burst_mask=batch["burst_mask"],
                            direction=batch["direction"],
                            token_mbm_labels=batch["token_mbm_labels"],
                            burst_mbm_mask=batch["burst_mbm_mask"],

                            token_loss_weight=args.token_loss_weight,
                            burst_loss_weight=args.burst_loss_weight,

                            nsp_pairs=nsp_pairs,
                            nsp_labels=nsp_labels,
                            nsp_loss_weight=args.nsp_loss_weight,

                            sodf_pairs=sodf_pairs,
                            sodf_labels=sodf_labels,
                            sodf_loss_weight=args.sodf_loss_weight,
                        )

                        loss = outputs["loss"]
                        if loss is not None:
                            loss_for_backward = loss / args.gradient_accumulation_steps
                else:
                    outputs = model(
                        src=batch["token_mbm_input"],
                        token_mask=batch["token_mask"],
                        burst_mask=batch["burst_mask"],
                        direction=batch["direction"],
                        token_mbm_labels=batch["token_mbm_labels"],
                        burst_mbm_mask=batch["burst_mbm_mask"],

                        token_loss_weight=args.token_loss_weight,
                        burst_loss_weight=args.burst_loss_weight,

                        nsp_pairs=nsp_pairs,
                        nsp_labels=nsp_labels,
                        nsp_loss_weight=args.nsp_loss_weight,

                        sodf_pairs=sodf_pairs,
                        sodf_labels=sodf_labels,
                        sodf_loss_weight=args.sodf_loss_weight,
                    )

                    loss = outputs["loss"]
                    if loss is not None:
                        loss_for_backward = loss / args.gradient_accumulation_steps

                if loss is None:
                    continue

                if use_amp:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            if not should_update:
                continue

            if use_amp:
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main_process() and global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1e-8)

                loss_token = outputs.get("loss_token_mbm", None)
                loss_burst = outputs.get("loss_burst_mbm", None)
                loss_nsp = outputs.get("loss_nsp", None)
                loss_sodf = outputs.get("loss_sodf", None)

                loss_token_val = loss_token.item() if loss_token is not None else 0.0
                loss_burst_val = loss_burst.item() if loss_burst is not None else 0.0
                loss_nsp_val = loss_nsp.item() if loss_nsp is not None else 0.0
                loss_sodf_val = loss_sodf.item() if loss_sodf is not None else 0.0

                nsp_pair_num = 0 if nsp_pairs is None else int(nsp_pairs.shape[0])
                sodf_pair_num = 0 if sodf_pairs is None else int(sodf_pairs.shape[0])

                print(
                    f"step={global_step} "
                    f"micro_step={micro_step + 1} "
                    f"loss={loss.item():.6f} "
                    f"token_mbm={loss_token_val:.6f} "
                    f"burst_mbm={loss_burst_val:.6f} "
                    f"nsp={loss_nsp_val:.6f} "
                    f"sodf={loss_sodf_val:.6f} "
                    f"nsp_pairs={nsp_pair_num} "
                    f"sodf_pairs={sodf_pair_num} "
                    f"steps/s={steps_per_sec:.3f}",
                    flush=True,
                )

                if loss_plotter is not None:
                    loss_plotter.update(
                        step=global_step,
                        loss=loss.item(),
                        token_mbm=loss_token_val,
                        burst_mbm=loss_burst_val,
                        nsp=loss_nsp_val,
                        sodf=loss_sodf_val,
                    )

            if is_main_process() and args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    args=args,
                    output_dir=args.output_dir,
                )

            if args.max_steps is not None and args.max_steps > 0:
                if global_step >= args.max_steps:
                    if is_main_process():
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            step=global_step,
                            args=args,
                            output_dir=args.output_dir,
                        )
                        print("[DONE] max_steps reached.", flush=True)
                    return

    if is_main_process():
        if loss_plotter is not None:
            loss_plotter.plot_all()
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=global_step,
            args=args,
            output_dir=args.output_dir,
        )

        print("[DONE] training finished.", flush=True)


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/pretrain/pretrain_data_0610_16*512*256/pt_manifest.jsonl",
        help=".pt 或 pt_manifest.jsonl",
    )

    parser.add_argument(
        "--dataset_info",
        type=str,
        default="datasets/pretrain/pretrain_data_0610_16*512*256/dataset_info.json",
        help="dataset_info.json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/pretrain_data_0610_16*512*256",
    )

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--burst_layers", type=int, default=4)
    parser.add_argument("--flow_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_direction_embedding", action="store_true")

    # mask
    parser.add_argument("--token_mlm_probability", type=float, default=0.15)
    parser.add_argument("--burst_mbm_probability", type=float, default=0.15)

    # loss weights
    parser.add_argument("--token_loss_weight", type=float, default=0.1)
    parser.add_argument("--burst_loss_weight", type=float, default=1.0)

    # NSP 和 SODF
    parser.add_argument(
        "--relation_task",
        type=str,
        default="sodf",
        choices=["none", "nsp", "sodf", "both"],
        help="关系预训练任务：none / nsp / sodf / both",
    )

    parser.add_argument("--relation_pairs", type=int, default=32)
    parser.add_argument("--nsp_positive_ratio", type=float, default=0.5)
    parser.add_argument("--nsp_loss_weight", type=float, default=1.0)
    parser.add_argument("--sodf_loss_weight", type=float, default=1.0)

    # train
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累计步数，用于增大等效 batch size")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--drop_last", action="store_true", default=True)

    # runtime
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dist_backend", type=str, default="auto", choices=["auto", "nccl", "gloo"], help="DDP backend")
    parser.add_argument("--ddp_timeout_minutes", type=int, default=60, help="DDP/NCCL collective timeout minutes")
    parser.add_argument("--find_unused_parameters", action="store_true", default=True, help="多任务头可能有未使用参数，先保持 True 更稳")
    parser.add_argument("--no_find_unused_parameters", action="store_false", dest="find_unused_parameters", help="关闭 find_unused_parameters")
    parser.add_argument("--shuffle", action="store_true", default=False, help="是否打乱训练数据。调试时建议 False，正式训练可开启。")
    parser.add_argument("--verbose_ranks", action="store_true", default=False, help="打印所有 DDP rank 的调试信息。默认只打印 rank0。")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # log/save
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=50000)

    parser.add_argument(
        "--plot_loss",
        action="store_true",
        default=True,
        help="是否保存 loss 曲线图",
    )

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")

    args.data = resolve_arg_path(args.data)
    args.dataset_info = resolve_arg_path(args.dataset_info)
    args.output_dir = resolve_arg_path(args.output_dir)

    train(args)


if __name__ == "__main__":
    main()
