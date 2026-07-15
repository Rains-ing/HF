import json
import os
import random
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from convert_checkpoint_to_bin import extract_model_state_dict, remove_module_prefix
from process_pretrain_data.hierarchical_dataset import load_dataset, load_dataset_info
from uer.models.hierarchical_burst_flow_classifier import HierarchicalBurstFlowClassifier


IGNORE_LABEL = -100


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class AnnotationRecord:
    burst_labels: list
    group_id: str | None = None


class AnnotationStore:
    """
    Burst semantic annotations stored separately from the large PT tensors.

    Each JSONL row must contain ``burst_labels`` and either ``sample_index``
    or ``flow_id``.  ``group_id`` is optional and is used to prevent donor
    selection from the same PCAP/device/session group in intervention tests.
    """

    def __init__(self, path=None):
        self.path = path
        self.by_index = {}
        self.by_flow_id = {}
        if path:
            self._load(path)

    def _load(self, path):
        with open(path, "r", encoding="utf-8") as reader:
            for line_number, line in enumerate(reader, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "burst_labels" not in item:
                    raise ValueError(
                        f"{path}:{line_number} missing burst_labels"
                    )
                record = AnnotationRecord(
                    burst_labels=list(item["burst_labels"]),
                    group_id=(
                        None
                        if item.get("group_id") is None
                        else str(item.get("group_id"))
                    ),
                )
                has_key = False
                if item.get("sample_index") is not None:
                    self.by_index[int(item["sample_index"])] = record
                    has_key = True
                if item.get("flow_id") not in (None, ""):
                    self.by_flow_id[str(item["flow_id"])] = record
                    has_key = True
                if not has_key:
                    raise ValueError(
                        f"{path}:{line_number} requires sample_index or flow_id"
                    )

    def get(self, index, sample):
        if index in self.by_index:
            return self.by_index[index]
        flow_id = sample.get("flow_id")
        if flow_id is not None and str(flow_id) in self.by_flow_id:
            return self.by_flow_id[str(flow_id)]
        return None


class SemanticDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, annotation_path=None):
        self.data_path = data_path
        self.dataset = load_dataset(data_path)
        self.annotations = AnnotationStore(annotation_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        record = self.annotations.get(index, sample)

        if record is not None:
            raw_labels = record.burst_labels
            group_id = record.group_id
        else:
            raw_labels = sample.get("burst_semantic_labels")
            group_id = sample.get("semantic_group_id", sample.get("pcap"))

        if raw_labels is None:
            if self.annotations.path:
                # Sparse sidecars are allowed: an omitted flow simply contributes
                # no labeled bursts to the semantic experiment.
                raw_labels = []
            else:
                raise ValueError(
                    f"No burst semantic labels for sample {index} in {self.data_path}. "
                    "Provide an annotation JSONL or add burst_semantic_labels to PT samples."
                )

        if torch.is_tensor(raw_labels):
            raw_labels = raw_labels.detach().cpu().tolist()

        sample["_sample_index"] = int(index)
        sample["_burst_semantic_raw"] = list(raw_labels)
        sample["_semantic_group_id"] = None if group_id is None else str(group_id)
        return sample


def _valid_raw_label(value):
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "ignore",
        "ignored",
        "none",
        "null",
        "unknown",
        "-100",
    }:
        return False
    try:
        if int(value) == IGNORE_LABEL and str(value).strip() == str(IGNORE_LABEL):
            return False
    except (TypeError, ValueError):
        pass
    return True


def _label_key(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).strip()


def build_label_map(datasets, explicit_path=None):
    if explicit_path:
        with open(explicit_path, "r", encoding="utf-8") as reader:
            mapping = json.load(reader)
        mapping = {str(k): int(v) for k, v in mapping.items()}
        values = sorted(mapping.values())
        if values != list(range(len(values))):
            raise ValueError("semantic label map values must be contiguous from 0")
        return mapping

    raw_values = set()
    for dataset in datasets:
        for index in range(len(dataset)):
            sample = dataset[index]
            for value in sample["_burst_semantic_raw"]:
                if _valid_raw_label(value):
                    raw_values.add(_label_key(value))

    if not raw_values:
        raise ValueError("No valid burst semantic labels found")

    def sort_key(value):
        try:
            return 0, int(value)
        except ValueError:
            return 1, value

    return {
        value: index
        for index, value in enumerate(sorted(raw_values, key=sort_key))
    }


def semantic_collate(samples, label_map):
    src = torch.stack([sample["src"] for sample in samples], dim=0)
    direction = torch.stack([sample["direction"] for sample in samples], dim=0)
    burst_mask = torch.stack([sample["burst_mask"] for sample in samples], dim=0)
    token_mask = torch.stack([sample["token_mask"] for sample in samples], dim=0)

    batch_size, max_bursts = burst_mask.shape
    semantic_labels = torch.full(
        (batch_size, max_bursts),
        IGNORE_LABEL,
        dtype=torch.long,
    )

    for row, sample in enumerate(samples):
        raw = sample["_burst_semantic_raw"][:max_bursts]
        for column, value in enumerate(raw):
            if not _valid_raw_label(value):
                continue
            key = _label_key(value)
            if key not in label_map:
                raise ValueError(f"Unknown burst semantic label: {value!r}")
            semantic_labels[row, column] = int(label_map[key])

    flow_labels = []
    for sample in samples:
        label = sample.get("label", torch.tensor(-1))
        flow_labels.append(label.long() if torch.is_tensor(label) else torch.tensor(int(label)))

    return {
        "src": src,
        "direction": direction,
        "burst_mask": burst_mask,
        "token_mask": token_mask,
        "semantic_labels": semantic_labels,
        "flow_labels": torch.stack(flow_labels).long(),
        "sample_indices": torch.tensor(
            [sample["_sample_index"] for sample in samples], dtype=torch.long
        ),
        "group_ids": [sample["_semantic_group_id"] for sample in samples],
    }


def make_loader(dataset, label_map, batch_size, num_workers, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(semantic_collate, label_map=label_map),
        drop_last=False,
    )


def add_model_arguments(parser):
    group = parser.add_argument_group("hierarchical model")
    group.add_argument("--dataset_info", required=True)
    group.add_argument("--checkpoint", required=True)
    group.add_argument("--hidden_size", type=int, default=256)
    group.add_argument("--burst_layers", type=int, default=4)
    group.add_argument("--flow_layers", type=int, default=2)
    group.add_argument("--num_heads", type=int, default=4)
    group.add_argument("--intermediate_size", type=int, default=1024)
    group.add_argument("--dropout", type=float, default=0.0)
    group.add_argument("--no_direction_embedding", action="store_true")
    group.add_argument("--num_flow_labels", type=int, default=2)


def build_and_load_model(args, device):
    dataset_info = load_dataset_info(args.dataset_info)
    special_ids = dataset_info["special_ids"]
    model = HierarchicalBurstFlowClassifier(
        vocab_size=int(dataset_info["vocab_size"]),
        num_labels=max(2, int(args.num_flow_labels)),
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

    obj = torch.load(args.checkpoint, map_location="cpu")
    state_dict = remove_module_prefix(extract_model_state_dict(obj))
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = sorted(set(state_dict) - set(compatible))
    missing, unexpected = model.load_state_dict(compatible, strict=False)

    required_prefixes = (
        "word_embedding.",
        "burst_encoder.",
        "flow_encoder.",
    )
    for prefix in required_prefixes:
        if not any(key.startswith(prefix) for key in compatible):
            raise ValueError(
                f"Checkpoint did not load any tensors for required prefix {prefix!r}. "
                "Check model dimensions and checkpoint type."
            )

    classifier_keys = [key for key in model_state if key.startswith("classifier.")]
    classifier_loaded = bool(classifier_keys) and all(
        key in compatible for key in classifier_keys
    )

    model.to(device)
    model.eval()
    return model, {
        "loaded_tensor_count": len(compatible),
        "skipped_tensor_count": len(skipped),
        "missing_tensor_count": len(missing),
        "unexpected_tensor_count": len(unexpected),
        "classifier_loaded": classifier_loaded,
    }


def move_batch(batch, device):
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def masked_mean_pool(flow_hidden, burst_mask):
    mask = burst_mask.float().unsqueeze(-1)
    return (flow_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def classification_metrics(predictions, labels, num_classes):
    predictions = predictions.detach().cpu().long()
    labels = labels.detach().cpu().long()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for truth, prediction in zip(labels, predictions):
        confusion[int(truth), int(prediction)] += 1

    per_class = []
    f1_values = []
    for class_id in range(num_classes):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum()) - true_positive
        false_negative = int(confusion[class_id, :].sum()) - true_positive
        support = int(confusion[class_id, :].sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        f1_values.append(f1)
        per_class.append(
            {
                "class_id": class_id,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    total = int(confusion.sum())
    accuracy = int(confusion.diag().sum()) / max(1, total)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / max(1, num_classes),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def save_json(path, obj):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(obj, writer, ensure_ascii=False, indent=2)


def inverse_label_map(label_map):
    return {str(value): key for key, value in label_map.items()}
