"""
Measure Contextual Semantic Gain (CSG).

The hierarchical encoder is frozen.  Two linear probes are trained with the
same protocol: one on local Burst-Encoder CLS vectors and one on contextual
Flow-Encoder burst vectors.  CSG is the difference between their test Macro-F1.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.burst_semantics.common import (
    IGNORE_LABEL,
    SemanticDataset,
    add_model_arguments,
    build_and_load_model,
    build_label_map,
    classification_metrics,
    inverse_label_map,
    make_loader,
    move_batch,
    resolve_device,
    save_json,
    set_seed,
)


@torch.no_grad()
def extract_embeddings(model, loader, device, max_bursts=None):
    local_vectors = []
    contextual_vectors = []
    labels = []
    sample_indices = []
    burst_indices = []

    collected = 0
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        local = model.encode_bursts(
            src=batch["src"],
            token_mask=batch["token_mask"],
        )
        contextual = model.encode_flow(
            burst_vectors=local,
            direction=batch["direction"],
            burst_mask=batch["burst_mask"],
        )

        valid = batch["burst_mask"].bool()
        valid &= batch["semantic_labels"].ne(IGNORE_LABEL)
        positions = torch.nonzero(valid, as_tuple=False)
        if positions.numel() == 0:
            continue

        if max_bursts is not None:
            remaining = int(max_bursts) - collected
            if remaining <= 0:
                break
            positions = positions[:remaining]

        rows = positions[:, 0]
        columns = positions[:, 1]
        local_vectors.append(local[rows, columns].detach().cpu())
        contextual_vectors.append(contextual[rows, columns].detach().cpu())
        labels.append(batch["semantic_labels"][rows, columns].detach().cpu())
        sample_indices.append(batch["sample_indices"][rows].detach().cpu())
        burst_indices.append(columns.detach().cpu())
        collected += int(positions.shape[0])

    if not labels:
        raise ValueError("No labeled bursts were extracted")

    return {
        "local": torch.cat(local_vectors, dim=0),
        "contextual": torch.cat(contextual_vectors, dim=0),
        "labels": torch.cat(labels, dim=0),
        "sample_indices": torch.cat(sample_indices, dim=0),
        "burst_indices": torch.cat(burst_indices, dim=0),
    }


class LinearProbe(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, features):
        return self.classifier(features)


def normalize_splits(train_features, valid_features, test_features):
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    return (
        (train_features - mean) / std,
        (valid_features - mean) / std,
        (test_features - mean) / std,
    )


def class_weights(labels, num_classes):
    counts = torch.bincount(labels, minlength=num_classes).float()
    if (counts == 0).any():
        missing = torch.nonzero(counts == 0, as_tuple=False).view(-1).tolist()
        raise ValueError(f"Training split has no examples for semantic classes {missing}")
    weights = counts.sum() / (num_classes * counts)
    return weights


@torch.no_grad()
def evaluate_probe(model, features, labels, batch_size, device, weights=None):
    model.eval()
    predictions = []
    total_loss = 0.0
    total_examples = 0
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        x = features[start:stop].to(device)
        y = labels[start:stop].to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits,
            y,
            weight=None if weights is None else weights.to(device),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_examples += int(y.numel())
        predictions.append(logits.argmax(dim=-1).cpu())

    metrics = classification_metrics(
        torch.cat(predictions, dim=0), labels, model.classifier.out_features
    )
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics


def train_probe(
    train_features,
    train_labels,
    valid_features,
    valid_labels,
    test_features,
    test_labels,
    num_classes,
    device,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    patience,
    balanced,
    seed,
):
    set_seed(seed)
    model = LinearProbe(train_features.shape[-1], num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    weights = class_weights(train_labels, num_classes) if balanced else None
    generator = torch.Generator().manual_seed(seed)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_macro_f1 = -1.0
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_labels), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            x = train_features[indices].to(device)
            y = train_labels[indices].to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits,
                y,
                weight=None if weights is None else weights.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        valid_metrics = evaluate_probe(
            model,
            valid_features,
            valid_labels,
            batch_size,
            device,
            weights,
        )
        history.append(
            {
                "epoch": epoch,
                "valid_macro_f1": valid_metrics["macro_f1"],
                "valid_accuracy": valid_metrics["accuracy"],
            }
        )
        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "valid": evaluate_probe(
            model, valid_features, valid_labels, batch_size, device, weights
        ),
        "test": evaluate_probe(
            model, test_features, test_labels, batch_size, device, weights
        ),
        "history": history,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate local and contextual burst semantics with linear probes."
    )
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--valid_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--train_annotations")
    parser.add_argument("--valid_annotations")
    parser.add_argument("--test_annotations")
    parser.add_argument("--semantic_label_map")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--extract_batch_size", type=int, default=32)
    parser.add_argument("--probe_batch_size", type=int, default=512)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--probe_patience", type=int, default=10)
    parser.add_argument("--no_balanced_loss", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_bursts_per_split", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    add_model_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_dataset = SemanticDataset(args.train_data, args.train_annotations)
    valid_dataset = SemanticDataset(args.valid_data, args.valid_annotations)
    test_dataset = SemanticDataset(args.test_data, args.test_annotations)
    label_map = build_label_map(
        [train_dataset, valid_dataset, test_dataset], args.semantic_label_map
    )
    num_classes = len(label_map)

    loaders = {
        "train": make_loader(
            train_dataset,
            label_map,
            args.extract_batch_size,
            args.num_workers,
        ),
        "valid": make_loader(
            valid_dataset,
            label_map,
            args.extract_batch_size,
            args.num_workers,
        ),
        "test": make_loader(
            test_dataset,
            label_map,
            args.extract_batch_size,
            args.num_workers,
        ),
    }

    model, checkpoint_report = build_and_load_model(args, device)
    embeddings = {
        name: extract_embeddings(
            model, loader, device, args.max_bursts_per_split
        )
        for name, loader in loaders.items()
    }

    results = {}
    probe_states = {}
    for representation in ("local", "contextual"):
        train_x, valid_x, test_x = normalize_splits(
            embeddings["train"][representation],
            embeddings["valid"][representation],
            embeddings["test"][representation],
        )
        probe_result = train_probe(
            train_x,
            embeddings["train"]["labels"],
            valid_x,
            embeddings["valid"]["labels"],
            test_x,
            embeddings["test"]["labels"],
            num_classes,
            device,
            args.probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            args.probe_patience,
            not args.no_balanced_loss,
            args.seed,
        )
        probe_states[representation] = probe_result.pop("state_dict")
        results[representation] = probe_result

    local_f1 = results["local"]["test"]["macro_f1"]
    contextual_f1 = results["contextual"]["test"]["macro_f1"]
    summary = {
        "metric": "Contextual Semantic Gain",
        "definition": "contextual_test_macro_f1 - local_test_macro_f1",
        "contextual_semantic_gain": contextual_f1 - local_f1,
        "local_test_macro_f1": local_f1,
        "contextual_test_macro_f1": contextual_f1,
        "label_map": label_map,
        "id_to_label": inverse_label_map(label_map),
        "split_labeled_bursts": {
            name: int(values["labels"].numel()) for name, values in embeddings.items()
        },
        "checkpoint": checkpoint_report,
        "results": results,
        "args": vars(args),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(os.path.join(args.output_dir, "contextual_semantic_gain.json"), summary)
    torch.save(probe_states, os.path.join(args.output_dir, "linear_probes.pt"))
    print(json.dumps({
        "local_test_macro_f1": local_f1,
        "contextual_test_macro_f1": contextual_f1,
        "contextual_semantic_gain": contextual_f1 - local_f1,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

