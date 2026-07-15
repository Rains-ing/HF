import os
import sys
import json
import time
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = CURRENT_DIR

if os.path.basename(PROJECT_ROOT).lower() in ["pre-train", "pretrain", "scripts"]:
    PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

sys.path.insert(0, PROJECT_ROOT)

if os.environ.get("HF_VERBOSE_IMPORT", "0") == "1":
    print("[PROJECT_ROOT]", PROJECT_ROOT)

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


def save_checkpoint(model, optimizer, step, args, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ckpt = {
        "step": step,
        "model": model.state_dict(),
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

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("[DEVICE]", device)

    dataset_info = load_dataset_info(args.dataset_info)
    dataset = load_dataset(args.data)

    collator = HierarchicalCollator(
        dataset_info=dataset_info,
        token_mlm_probability=args.token_mlm_probability,
        burst_mbm_probability=args.burst_mbm_probability,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=args.drop_last,
    )

    model = build_model(args, dataset_info)
    model.to(device)

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

    print("[MODEL]")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"total params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")

    print("[DATA]")
    print(f"samples: {len(dataset)}")
    print(f"batch_size: {args.batch_size}")
    print(f"steps per epoch: {len(dataloader)}")

    print("[TASK]")
    print(f"relation_task: {args.relation_task}")
    print(f"token_mlm_probability: {args.token_mlm_probability}")
    print(f"burst_mbm_probability: {args.burst_mbm_probability}")
    print(f"relation_pairs: {args.relation_pairs}")

    global_step = 0
    start_time = time.time()

    loss_plotter = None
    if args.plot_loss:
        loss_plotter = LossPlotter(output_dir=args.output_dir)
        print("[PLOT]")
        print(f"loss csv: {loss_plotter.csv_path}")
        for name, path in loss_plotter.png_paths.items():
            print(f"{name} png: {path}")

    model.train()

    for epoch in range(1, args.epochs + 1):
        print(f"\n[EPOCH {epoch}/{args.epochs}]")

        for batch in dataloader:
            global_step += 1

            batch = move_batch_to_device(batch, device)

            # 动态构造 NSP / SODF 关系任务
            nsp_pairs, nsp_labels, sodf_pairs, sodf_labels = build_relation_tasks(
                args=args,
                batch=batch,
            )

            optimizer.zero_grad(set_to_none=True)

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

            if loss is None:
                continue

            if use_amp:
                scaler.scale(loss).backward()

                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

                optimizer.step()

            if global_step % args.log_every == 0:
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
                    f"loss={loss.item():.6f} "
                    f"token_mbm={loss_token_val:.6f} "
                    f"burst_mbm={loss_burst_val:.6f} "
                    f"nsp={loss_nsp_val:.6f} "
                    f"sodf={loss_sodf_val:.6f} "
                    f"nsp_pairs={nsp_pair_num} "
                    f"sodf_pairs={sodf_pair_num} "
                    f"steps/s={steps_per_sec:.3f}"
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

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    args=args,
                    output_dir=args.output_dir,
                )

            if args.max_steps is not None and args.max_steps > 0:
                if global_step >= args.max_steps:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=global_step,
                        args=args,
                        output_dir=args.output_dir,
                    )
                    print("[DONE] max_steps reached.")
                    return
    if loss_plotter is not None:
        loss_plotter.plot_all()
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=global_step,
        args=args,
        output_dir=args.output_dir,
    )

    print("[DONE] training finished.")


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--data",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/pretrain_data/pt_manifest_min_data.jsonl",
        help=".pt 或 pt_manifest.jsonl",
    )

    parser.add_argument(
        "--dataset_info",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/pretrain_data/dataset_info.json",
        help="dataset_info.json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/models/pretrain_model",
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
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--drop_last", action="store_true", default=True)

    # runtime
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # log/save
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=80000)

    parser.add_argument(
        "--plot_loss",
        action="store_true",
        default=True,
        help="是否保存 loss 曲线图",
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
