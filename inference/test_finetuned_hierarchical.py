import os
import sys
import json
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader


# =========================
# Project path
# =========================
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


def infer_num_labels(dataset):
    labels = set()

    for sample in dataset.samples:
        label = sample["label"]

        if torch.is_tensor(label):
            label = int(label.item())
        else:
            label = int(label)

        labels.add(label)

    if len(labels) == 0:
        raise ValueError("test 数据集中没有 label")

    return max(labels) + 1, sorted(labels)


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


def compute_metrics(preds, labels, num_labels):
    """
    计算：
        Accuracy
        Macro Precision
        Macro Recall
        Macro F1
        Per-class Precision / Recall / F1
        Confusion Matrix

    confusion_matrix:
        行 = 真实标签 True Label
        列 = 预测标签 Predicted Label
    """

    preds = preds.detach().cpu()
    labels = labels.detach().cpu()

    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)

    for y_true, y_pred in zip(labels, preds):
        confusion[int(y_true), int(y_pred)] += 1

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()

    accuracy = correct / total if total > 0 else 0.0

    per_class = []
    precisions = []
    recalls = []
    f1s = []

    for i in range(num_labels):
        tp = confusion[i, i].item()
        fp = confusion[:, i].sum().item() - tp
        fn = confusion[i, :].sum().item() - tp
        support = confusion[i, :].sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class.append({
            "label": i,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    macro_precision = sum(precisions) / num_labels
    macro_recall = sum(recalls) / num_labels
    macro_f1 = sum(f1s) / num_labels

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


@torch.no_grad()
def test(model, dataloader, device, num_labels, log_every=100):
    model.eval()

    total_loss = 0.0
    total_steps = 0

    all_preds = []
    all_labels = []

    for step, batch in enumerate(dataloader, start=1):
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

        if log_every > 0 and step % log_every == 0:
            print(f"[TEST] step={step} batch_loss={loss.item():.6f}")

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(
        preds=all_preds,
        labels=all_labels,
        num_labels=num_labels,
    )

    metrics["loss"] = total_loss / max(total_steps, 1)

    return metrics, all_preds, all_labels


def print_metrics(metrics):
    print("\n==================== TEST RESULT ====================")
    print(f"Test Loss              : {metrics['loss']:.6f}")
    print(f"Accuracy               : {metrics['accuracy']:.4f}")
    print(f"Macro Precision        : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall           : {metrics['macro_recall']:.4f}")
    print(f"Macro F1-score         : {metrics['macro_f1']:.4f}")
    print("=====================================================")

    print("\n指标说明：")
    print("Accuracy        = 所有样本中预测正确的比例")
    print("Precision       = 预测为某类的样本中，有多少是真的该类")
    print("Recall          = 某个真实类别中，有多少被模型找出来")
    print("F1-score        = Precision 和 Recall 的调和平均")
    print("Macro Precision = 各类别 Precision 的平均值")
    print("Macro Recall    = 各类别 Recall 的平均值")
    print("Macro F1-score  = 各类别 F1-score 的平均值")

    print("\n==================== PER-CLASS METRICS ====================")
    print(f"{'Label':<10}{'Support':<12}{'Precision':<15}{'Recall':<15}{'F1-score':<15}")
    print("-" * 67)

    for item in metrics["per_class"]:
        print(
            f"{item['label']:<10}"
            f"{item['support']:<12}"
            f"{item['precision']:<15.4f}"
            f"{item['recall']:<15.4f}"
            f"{item['f1']:<15.4f}"
        )

    print("===========================================================")

    print("\n==================== CONFUSION MATRIX ====================")
    print("行表示真实标签 True Label，列表示预测标签 Predicted Label")
    print(metrics["confusion_matrix"])
    print("==========================================================")


def save_results(metrics, preds, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    cm_path = os.path.join(output_dir, "test_confusion_matrix.txt")
    pred_path = os.path.join(output_dir, "test_predictions.tsv")
    report_path = os.path.join(output_dir, "test_report.txt")

    metrics_to_save = dict(metrics)
    metrics_to_save["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)

    with open(cm_path, "w", encoding="utf-8") as f:
        f.write("行表示真实标签 True Label，列表示预测标签 Predicted Label\n")
        f.write(str(metrics["confusion_matrix"]))
        f.write("\n")

    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("index\tlabel\tprediction\n")

        for i, (label, pred) in enumerate(zip(labels.tolist(), preds.tolist())):
            f.write(f"{i}\t{label}\t{pred}\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================== TEST RESULT ====================\n")
        f.write(f"Test Loss              : {metrics['loss']:.6f}\n")
        f.write(f"Accuracy               : {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision        : {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall           : {metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1-score         : {metrics['macro_f1']:.4f}\n")
        f.write("=====================================================\n\n")

        f.write("指标说明：\n")
        f.write("Accuracy        = 所有样本中预测正确的比例\n")
        f.write("Precision       = 预测为某类的样本中，有多少是真的该类\n")
        f.write("Recall          = 某个真实类别中，有多少被模型找出来\n")
        f.write("F1-score        = Precision 和 Recall 的调和平均\n")
        f.write("Macro Precision = 各类别 Precision 的平均值\n")
        f.write("Macro Recall    = 各类别 Recall 的平均值\n")
        f.write("Macro F1-score  = 各类别 F1-score 的平均值\n\n")

        f.write("==================== PER-CLASS METRICS ====================\n")
        f.write(f"{'Label':<10}{'Support':<12}{'Precision':<15}{'Recall':<15}{'F1-score':<15}\n")
        f.write("-" * 67 + "\n")

        for item in metrics["per_class"]:
            f.write(
                f"{item['label']:<10}"
                f"{item['support']:<12}"
                f"{item['precision']:<15.4f}"
                f"{item['recall']:<15.4f}"
                f"{item['f1']:<15.4f}\n"
            )

        f.write("===========================================================\n\n")

        f.write("==================== CONFUSION MATRIX ====================\n")
        f.write("行表示真实标签 True Label，列表示预测标签 Predicted Label\n")
        f.write(str(metrics["confusion_matrix"]))
        f.write("\n==========================================================\n")

    print(f"[SAVE] {metrics_path}")
    print(f"[SAVE] {cm_path}")
    print(f"[SAVE] {pred_path}")
    print(f"[SAVE] {report_path}")


def main():
    parser = argparse.ArgumentParser()

    # =========================
    # Data / model paths
    # =========================
    parser.add_argument(
        "--test_path",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/finetuning-data_reset_0527_ISCX-VPN-Service_test/test.pt",
        help="测试集 test.pt",
    )

    parser.add_argument(
        "--dataset_info",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/pretrain_data/dataset_info.json",
        help="dataset_info.json",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/models/finetuned_model_0527_50000/finetuned_model.bin",
        help="训练好的 finetuned_model.bin",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/result/0527",
        help="测试结果保存目录",
    )

    # =========================
    # Model config
    # 注意：这些参数必须和微调训练时一致
    # =========================
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--burst_layers", type=int, default=4)
    parser.add_argument("--flow_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_direction_embedding", action="store_true")

    # =========================
    # Runtime
    # =========================
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_every", type=int, default=100)

    args = parser.parse_args()

    # =========================
    # Device
    # =========================
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("[DEVICE]", device)

    # =========================
    # Load dataset info and test data
    # =========================
    dataset_info = load_dataset_info(args.dataset_info)

    test_dataset = FinetunePTDataset(args.test_path)

    print("\n[TEST DATA]")
    print("test_path:", args.test_path)
    print("samples:", len(test_dataset))
    print("label distribution:", count_labels(test_dataset))

    num_labels, label_list = infer_num_labels(test_dataset)

    print("\n[LABEL INFO]")
    print("labels:", label_list)
    print("num_labels:", num_labels)

    collator = FinetuneCollator()

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=False,
    )

    # =========================
    # Build model
    # =========================
    model = build_model(
        args=args,
        dataset_info=dataset_info,
        num_labels=num_labels,
    )

    # =========================
    # Load finetuned model
    # =========================
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    print(f"\n[LOAD MODEL] {args.model_path}")

    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.to(device)

    # =========================
    # Test
    # =========================
    metrics, preds, labels = test(
        model=model,
        dataloader=test_loader,
        device=device,
        num_labels=num_labels,
        log_every=args.log_every,
    )

    print_metrics(metrics)

    save_results(
        metrics=metrics,
        preds=preds,
        labels=labels,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()