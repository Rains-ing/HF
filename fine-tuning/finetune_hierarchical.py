import os
import sys
import json
import time
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# 项目根目录
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

print("[PROJECT_ROOT]", PROJECT_ROOT)

from process_fine_tuning.finetune_dataset import (
    FinetunePTDataset,
    FinetuneCollator,
)

from process_pretrain_data.hierarchical_dataset import load_dataset_info

from uer.models.hierarchical_burst_flow_classifier import (
    HierarchicalBurstFlowClassifier,
)
from uer.finetune_lr_scheduler import (
    LR_SCHEDULERS,
    build_lr_schedule_config,
    get_learning_rate,
    set_optimizer_learning_rate,
)


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    new_batch = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v

    return new_batch


def count_labels(dataset):
    counter = Counter()

    for sample in dataset.samples:
        label = sample["label"]
        if torch.is_tensor(label):
            label = int(label.item())
        else:
            label = int(label)
        counter[label] += 1

    return dict(sorted(counter.items()))


def infer_num_labels(*datasets):
    labels = set()

    for dataset in datasets:
        for sample in dataset.samples:
            label = sample["label"]
            if torch.is_tensor(label):
                label = int(label.item())
            else:
                label = int(label)
            labels.add(label)

    if len(labels) == 0:
        raise ValueError("没有找到任何 label")

    max_label = max(labels)

    return max_label + 1, sorted(labels)


def filter_state_dict_by_shape(model, state_dict):
    """
    只加载 key 存在且 shape 一致的权重。
    这样可以自动跳过：
        token_mbm_head
        burst_mbm_head
        nsp_head
        sodf_head
        classifier
        以及 shape 不匹配的位置编码
    """
    model_state = model.state_dict()

    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    return filtered, skipped


def load_pretrained_weights(model, pretrained_path):
    if pretrained_path is None or pretrained_path == "":
        print("[PRETRAIN] no pretrained weights loaded.")
        return

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(pretrained_path)

    print(f"[LOAD PRETRAIN] {pretrained_path}")

    obj = torch.load(pretrained_path, map_location="cpu")

    if isinstance(obj, dict) and "model" in obj:
        state_dict = obj["model"]
    else:
        state_dict = obj

    filtered, skipped = filter_state_dict_by_shape(model, state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)

    print(f"[PRETRAIN] loaded tensors: {len(filtered)}")
    print(f"[PRETRAIN] skipped tensors: {len(skipped)}")
    print(f"[PRETRAIN] missing_keys: {len(missing_keys)}")
    print(f"[PRETRAIN] unexpected_keys: {len(unexpected_keys)}")

    print("\n[SKIPPED KEYS SAMPLE]")
    for k in skipped[:30]:
        print(k)

    print("\n[MISSING KEYS SAMPLE]")
    for k in missing_keys[:30]:
        print(k)


def compute_metrics(preds, labels, num_labels):
    """
    不依赖 sklearn，直接计算：
        accuracy
        per-class precision / recall / f1
        macro precision / recall / f1
        confusion matrix
    """
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()

    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)

    for y_true, y_pred in zip(labels, preds):
        confusion[int(y_true), int(y_pred)] += 1

    correct = confusion.diag().sum().item()
    total = confusion.sum().item()

    acc = correct / total if total > 0 else 0.0

    per_class = []

    precisions = []
    recalls = []
    f1s = []

    for i in range(num_labels):
        tp = confusion[i, i].item()
        fp = confusion[:, i].sum().item() - tp
        fn = confusion[i, :].sum().item() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class.append({
            "label": i,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    macro_precision = sum(precisions) / num_labels
    macro_recall = sum(recalls) / num_labels
    macro_f1 = sum(f1s) / num_labels

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, num_labels):
    model.eval()

    total_loss = 0.0
    total_steps = 0

    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        outputs = model(
            src=batch["src"],
            token_mask=batch["token_mask"],
            burst_mask=batch["burst_mask"],
            direction=batch["direction"],
            labels=batch["label"],
        )

        loss = outputs["loss"]
        logits = outputs["logits"]

        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item()
        total_steps += 1

        all_preds.append(preds.detach().cpu())
        all_labels.append(batch["label"].detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(
        preds=all_preds,
        labels=all_labels,
        num_labels=num_labels,
    )

    metrics["loss"] = total_loss / max(total_steps, 1)

    model.train()

    return metrics


def save_finetuned_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "finetuned_model.bin")
    torch.save(model.state_dict(), path)

    print(f"[SAVE FINETUNED MODEL] {path}")


import os
import json
import numpy as np
import torch


def save_metrics(metrics, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{name}_metrics.json")
    cm_path = os.path.join(output_dir, f"{name}_confusion_matrix.txt")
    cm_csv_path = os.path.join(output_dir, f"{name}_confusion_matrix.csv")

    save_obj = dict(metrics)

    cm = metrics["confusion_matrix"]

    # 如果是 torch.Tensor，转成 numpy
    if isinstance(cm, torch.Tensor):
        cm_np = cm.detach().cpu().numpy()
    else:
        cm_np = np.array(cm)

    # JSON 中保存为完整二维 list
    save_obj["confusion_matrix"] = cm_np.tolist()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)

    # txt 中完整保存，不省略
    with open(cm_path, "w", encoding="utf-8") as f:
        np.savetxt(f, cm_np, fmt="%d")

    # 额外保存 csv，方便 Excel 打开
    np.savetxt(cm_csv_path, cm_np, fmt="%d", delimiter=",")

    print(f"[SAVE METRICS] {json_path}")
    print(f"[SAVE CONFUSION MATRIX TXT] {cm_path}")
    print(f"[SAVE CONFUSION MATRIX CSV] {cm_csv_path}")


def build_model(args, dataset_info, num_labels):
    special_ids = dataset_info["special_ids"]

    model = HierarchicalBurstFlowClassifier(
        vocab_size=int(dataset_info["vocab_size"]),
        num_labels=num_labels,
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
        direction_vocab_size=3,
        use_direction_embedding=not args.no_direction_embedding,
    )

    return model


def train(args):
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("[DEVICE]", device)

    dataset_info = load_dataset_info(args.dataset_info)

    train_dataset = FinetunePTDataset(args.train_path)
    valid_dataset = FinetunePTDataset(args.valid_path)
    test_dataset = FinetunePTDataset(args.test_path)

    print("[LABEL DISTRIBUTION]")
    print("train:", count_labels(train_dataset))
    print("valid:", count_labels(valid_dataset))
    print("test :", count_labels(test_dataset))

    num_labels, label_list = infer_num_labels(
        train_dataset,
        valid_dataset,
        test_dataset,
    )

    print("[LABEL INFO]")
    print("labels:", label_list)
    print("num_labels:", num_labels)

    collator = FinetuneCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=args.drop_last,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=False,
    )

    model = build_model(
        args=args,
        dataset_info=dataset_info,
        num_labels=num_labels,
    )

    load_pretrained_weights(
        model=model,
        pretrained_path=args.pretrained_model_path,
    )

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    lr_schedule = build_lr_schedule_config(
        args=args,
        steps_per_epoch=len(train_loader),
    )

    print("[LR SCHEDULER]")
    print(f"type: {lr_schedule['scheduler_name']}")
    print(f"peak_lr: {lr_schedule['peak_lr']:.8g}")
    print(f"min_lr: {lr_schedule['min_lr']:.8g}")
    print(f"warmup_ratio: {lr_schedule['warmup_ratio']:.4f}")
    print(f"warmup_steps: {lr_schedule['warmup_steps']}")
    print(f"total_steps: {lr_schedule['total_steps']}")

    use_amp = args.fp16 and device.type == "cuda"

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("[MODEL]")
    print(f"total params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")

    global_step = 0
    best_valid_f1 = -1.0

    model.train()

    for epoch in range(1, args.epochs + 1):
        print(f"\n[EPOCH {epoch}/{args.epochs}]")

        start_time = time.time()

        for batch in train_loader:
            global_step += 1

            current_lr = get_learning_rate(
                current_step=global_step - 1,
                total_steps=lr_schedule["total_steps"],
                peak_lr=lr_schedule["peak_lr"],
                min_lr=lr_schedule["min_lr"],
                warmup_steps=lr_schedule["warmup_steps"],
                scheduler_name=lr_schedule["scheduler_name"],
            )
            set_optimizer_learning_rate(optimizer, current_lr)

            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                try:
                    autocast_ctx = torch.amp.autocast("cuda", enabled=True)
                except Exception:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=True)

                with autocast_ctx:
                    outputs = model(
                        src=batch["src"],
                        token_mask=batch["token_mask"],
                        burst_mask=batch["burst_mask"],
                        direction=batch["direction"],
                        labels=batch["label"],
                    )

                    loss = outputs["loss"]
            else:
                outputs = model(
                    src=batch["src"],
                    token_mask=batch["token_mask"],
                    burst_mask=batch["burst_mask"],
                    direction=batch["direction"],
                    labels=batch["label"],
                )

                loss = outputs["loss"]

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

                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)
                train_acc = (preds == batch["label"]).float().mean().item()

                print(
                    f"step={global_step} "
                    f"loss={loss.item():.6f} "
                    f"train_acc={train_acc:.4f} "
                    f"lr={current_lr:.8g} "
                    f"steps/s={steps_per_sec:.3f}"
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                print("[STOP] max_steps reached.")
                break

        valid_metrics = evaluate(
            model=model,
            dataloader=valid_loader,
            device=device,
            num_labels=num_labels,
        )

        print(
            f"[VALID] epoch={epoch} "
            f"loss={valid_metrics['loss']:.6f} "
            f"acc={valid_metrics['accuracy']:.4f} "
            f"macro_p={valid_metrics['macro_precision']:.4f} "
            f"macro_r={valid_metrics['macro_recall']:.4f} "
            f"macro_f1={valid_metrics['macro_f1']:.4f}"
        )

        save_metrics(
            metrics=valid_metrics,
            output_dir=args.output_dir,
            name=f"valid_epoch_{epoch}",
        )

        if valid_metrics["macro_f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["macro_f1"]
            save_finetuned_model(model, args.output_dir)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    print("\n[TEST] loading best model...")

    best_model_path = os.path.join(args.output_dir, "finetuned_model.bin")
    state_dict = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        num_labels=num_labels,
    )

    print(
        f"[TEST] "
        f"loss={test_metrics['loss']:.6f} "
        f"acc={test_metrics['accuracy']:.4f} "
        f"macro_p={test_metrics['macro_precision']:.4f} "
        f"macro_r={test_metrics['macro_recall']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )

    print("[TEST CONFUSION MATRIX]")
    print(test_metrics["confusion_matrix"])

    save_metrics(
        metrics=test_metrics,
        output_dir=args.output_dir,
        name="test",
    )


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--train_path",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/finetuning-data_reset_0604_cstnet-tls_only_de_ip/train.pt",
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/finetuning-data_reset_0604_cstnet-tls_only_de_ip/valid.pt",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/finetuning-data_reset_0604_cstnet-tls_only_de_ip/test.pt",
    )

    parser.add_argument(
        "--dataset_info",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/dataset_info.json",
    )

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/models/model_bin/pretrained_model_0602_50000.bin",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/models/finetuned_model_0602_36000_cstnet-tls_20_only_de_ip",
    )

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--burst_layers", type=int, default=4)
    parser.add_argument("--flow_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_direction_embedding", action="store_true")

    # train
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--lr_scheduler",
        choices=LR_SCHEDULERS,
        default="cosine",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1)

    # runtime
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # log
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
