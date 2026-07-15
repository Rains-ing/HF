"""
Measure Semantic Intervention Gap (SIG).

For a labeled target burst, replace its local vector with a donor vector of
the same semantic class and with a donor of a different semantic class.  Both
donors are direction matched; token-count matching is enabled by default.
SIG is the paired difference between the two induced flow-embedding drifts.
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
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
    inverse_label_map,
    make_loader,
    masked_mean_pool,
    move_batch,
    resolve_device,
    save_json,
    set_seed,
)


def reservoir_add(bank, seen, key, entry, capacity, rng):
    seen[key] += 1
    bucket = bank[key]
    if len(bucket) < capacity:
        bucket.append(entry)
        return
    replacement = rng.randrange(seen[key])
    if replacement < capacity:
        bucket[replacement] = entry


@torch.no_grad()
def build_donor_bank(model, loader, device, capacity, seed):
    rng = random.Random(seed)
    bank = defaultdict(list)
    seen = defaultdict(int)

    for batch in loader:
        batch = move_batch(batch, device)
        local = model.encode_bursts(batch["src"], batch["token_mask"])
        valid = batch["burst_mask"].bool()
        valid &= batch["semantic_labels"].ne(IGNORE_LABEL)
        for row, column in torch.nonzero(valid, as_tuple=False).tolist():
            label = int(batch["semantic_labels"][row, column])
            direction = int(batch["direction"][row, column])
            if direction not in (0, 1):
                continue
            entry = {
                "vector": local[row, column].detach().cpu(),
                "label": label,
                "direction": direction,
                "token_count": int(batch["token_mask"][row, column].sum()),
                "sample_index": int(batch["sample_indices"][row]),
                "burst_index": int(column),
                "group_id": batch["group_ids"][row],
            }
            reservoir_add(
                bank,
                seen,
                (label, direction),
                entry,
                capacity,
                rng,
            )
    return bank, dict(seen)


def valid_donor(entry, sample_index, group_id, exclude_same_group):
    if entry["sample_index"] == sample_index:
        return False
    if (
        exclude_same_group
        and group_id is not None
        and entry["group_id"] is not None
        and entry["group_id"] == group_id
    ):
        return False
    return True


def choose_candidate(candidates, target_token_count, match_token_count, rng):
    if not candidates:
        return None
    if not match_token_count:
        return rng.choice(candidates)

    distances = [
        abs(int(entry["token_count"]) - int(target_token_count))
        for entry in candidates
    ]
    minimum = min(distances)
    closest = [
        entry for entry, distance in zip(candidates, distances) if distance == minimum
    ]
    return rng.choice(closest)


def select_donors(
    bank,
    target_label,
    target_direction,
    target_token_count,
    sample_index,
    group_id,
    exclude_same_group,
    match_token_count,
    rng,
):
    same_candidates = [
        entry
        for entry in bank.get((target_label, target_direction), [])
        if valid_donor(entry, sample_index, group_id, exclude_same_group)
    ]
    different_candidates = []
    for (label, direction), entries in bank.items():
        if label == target_label or direction != target_direction:
            continue
        different_candidates.extend(
            entry
            for entry in entries
            if valid_donor(entry, sample_index, group_id, exclude_same_group)
        )

    return (
        choose_candidate(
            same_candidates, target_token_count, match_token_count, rng
        ),
        choose_candidate(
            different_candidates, target_token_count, match_token_count, rng
        ),
    )


def bootstrap_mean_ci(values, samples, confidence, seed):
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return [None, None]
    if tensor.numel() == 1 or samples <= 0:
        value = float(tensor.mean())
        return [value, value]

    generator = torch.Generator().manual_seed(seed)
    means = []
    chunk_size = min(512, samples)
    remaining = samples
    while remaining > 0:
        current = min(chunk_size, remaining)
        indices = torch.randint(
            0,
            tensor.numel(),
            (current, tensor.numel()),
            generator=generator,
        )
        means.append(tensor[indices].mean(dim=1))
        remaining -= current
    means = torch.cat(means).sort().values
    alpha = (1.0 - confidence) / 2.0
    lower = int(alpha * (len(means) - 1))
    upper = int((1.0 - alpha) * (len(means) - 1))
    return [float(means[lower]), float(means[upper])]


def summarize_records(records, num_classes, confidence, bootstrap_samples, seed):
    same = [record["same_semantic_distance"] for record in records]
    different = [record["different_semantic_distance"] for record in records]
    gaps = [record["semantic_intervention_gap"] for record in records]

    per_class = {}
    for class_id in range(num_classes):
        class_records = [
            record for record in records if record["target_semantic_id"] == class_id
        ]
        class_gaps = [record["semantic_intervention_gap"] for record in class_records]
        per_class[str(class_id)] = {
            "count": len(class_records),
            "mean_gap": (
                statistics.fmean(class_gaps) if class_gaps else None
            ),
        }

    summary = {
        "case_count": len(records),
        "mean_same_semantic_distance": statistics.fmean(same),
        "mean_different_semantic_distance": statistics.fmean(different),
        "semantic_intervention_gap": statistics.fmean(gaps),
        "median_semantic_intervention_gap": statistics.median(gaps),
        "positive_gap_fraction": sum(value > 0 for value in gaps) / len(gaps),
        "bootstrap_confidence": confidence,
        "semantic_intervention_gap_ci": bootstrap_mean_ci(
            gaps, bootstrap_samples, confidence, seed
        ),
        "per_class": per_class,
    }

    if "same_prediction_changed" in records[0]:
        summary.update(
            {
                "same_semantic_prediction_flip_rate": statistics.fmean(
                    record["same_prediction_changed"] for record in records
                ),
                "different_semantic_prediction_flip_rate": statistics.fmean(
                    record["different_prediction_changed"] for record in records
                ),
            }
        )
    return summary


@torch.no_grad()
def evaluate_interventions(
    model,
    loader,
    donor_bank,
    device,
    targets_per_flow,
    max_cases,
    exclude_same_group,
    match_token_count,
    classifier_loaded,
    seed,
):
    rng = random.Random(seed)
    records = []
    skipped_no_same = 0
    skipped_no_different = 0

    for batch in loader:
        batch = move_batch(batch, device)
        local = model.encode_bursts(batch["src"], batch["token_mask"])
        clean_contextual = model.encode_flow(
            local, batch["direction"], batch["burst_mask"]
        )
        clean_flow = masked_mean_pool(clean_contextual, batch["burst_mask"])

        cases = []
        for row in range(local.shape[0]):
            valid_columns = torch.nonzero(
                batch["burst_mask"][row].bool()
                & batch["semantic_labels"][row].ne(IGNORE_LABEL),
                as_tuple=False,
            ).view(-1).tolist()
            rng.shuffle(valid_columns)
            for column in valid_columns[:targets_per_flow]:
                label = int(batch["semantic_labels"][row, column])
                direction = int(batch["direction"][row, column])
                token_count = int(batch["token_mask"][row, column].sum())
                sample_index = int(batch["sample_indices"][row])
                group_id = batch["group_ids"][row]
                same_donor, different_donor = select_donors(
                    donor_bank,
                    label,
                    direction,
                    token_count,
                    sample_index,
                    group_id,
                    exclude_same_group,
                    match_token_count,
                    rng,
                )
                if same_donor is None:
                    skipped_no_same += 1
                    continue
                if different_donor is None:
                    skipped_no_different += 1
                    continue
                cases.append(
                    {
                        "row": row,
                        "column": column,
                        "label": label,
                        "direction": direction,
                        "token_count": token_count,
                        "sample_index": sample_index,
                        "group_id": group_id,
                        "same": same_donor,
                        "different": different_donor,
                    }
                )
                if max_cases is not None and len(records) + len(cases) >= max_cases:
                    break
            if max_cases is not None and len(records) + len(cases) >= max_cases:
                break

        if not cases:
            if max_cases is not None and len(records) >= max_cases:
                break
            continue

        same_inputs = []
        different_inputs = []
        directions = []
        burst_masks = []
        clean_vectors = []
        for case in cases:
            row = case["row"]
            column = case["column"]
            same_input = local[row].clone()
            different_input = local[row].clone()
            same_input[column] = case["same"]["vector"].to(device)
            different_input[column] = case["different"]["vector"].to(device)
            same_inputs.append(same_input)
            different_inputs.append(different_input)
            directions.append(batch["direction"][row])
            burst_masks.append(batch["burst_mask"][row])
            clean_vectors.append(clean_flow[row])

        same_inputs = torch.stack(same_inputs)
        different_inputs = torch.stack(different_inputs)
        directions = torch.stack(directions)
        burst_masks = torch.stack(burst_masks)
        clean_vectors = torch.stack(clean_vectors)

        same_flow = masked_mean_pool(
            model.encode_flow(same_inputs, directions, burst_masks), burst_masks
        )
        different_flow = masked_mean_pool(
            model.encode_flow(different_inputs, directions, burst_masks), burst_masks
        )
        same_distance = 1.0 - F.cosine_similarity(clean_vectors, same_flow, dim=-1)
        different_distance = 1.0 - F.cosine_similarity(
            clean_vectors, different_flow, dim=-1
        )

        if classifier_loaded:
            clean_predictions = model.classifier(clean_vectors).argmax(dim=-1)
            same_predictions = model.classifier(same_flow).argmax(dim=-1)
            different_predictions = model.classifier(different_flow).argmax(dim=-1)

        for index, case in enumerate(cases):
            record = {
                "sample_index": case["sample_index"],
                "group_id": case["group_id"],
                "target_burst_index": case["column"],
                "target_semantic_id": case["label"],
                "direction": case["direction"],
                "target_token_count": case["token_count"],
                "same_donor_sample_index": case["same"]["sample_index"],
                "same_donor_burst_index": case["same"]["burst_index"],
                "same_donor_token_count": case["same"]["token_count"],
                "different_donor_semantic_id": case["different"]["label"],
                "different_donor_sample_index": case["different"]["sample_index"],
                "different_donor_burst_index": case["different"]["burst_index"],
                "different_donor_token_count": case["different"]["token_count"],
                "same_semantic_distance": float(same_distance[index]),
                "different_semantic_distance": float(different_distance[index]),
                "semantic_intervention_gap": float(
                    different_distance[index] - same_distance[index]
                ),
            }
            if classifier_loaded:
                record["same_prediction_changed"] = int(
                    same_predictions[index] != clean_predictions[index]
                )
                record["different_prediction_changed"] = int(
                    different_predictions[index] != clean_predictions[index]
                )
            records.append(record)

        if max_cases is not None and len(records) >= max_cases:
            records = records[:max_cases]
            break

    if not records:
        raise ValueError(
            "No valid intervention cases. Check class coverage, directions, groups, "
            "and semantic annotations."
        )
    return records, {
        "skipped_no_same_semantic_donor": skipped_no_same,
        "skipped_no_different_semantic_donor": skipped_no_different,
    }


def write_records(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic burst replacements in the flow embedding space."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--semantic_label_map")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--targets_per_flow", type=int, default=2)
    parser.add_argument("--max_cases", type=int)
    parser.add_argument("--donors_per_label_direction", type=int, default=1000)
    parser.add_argument("--allow_same_group", action="store_true")
    parser.add_argument("--no_token_count_matching", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    add_model_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = SemanticDataset(args.data, args.annotations)
    label_map = build_label_map([dataset], args.semantic_label_map)
    loader = make_loader(
        dataset, label_map, args.batch_size, args.num_workers, shuffle=False
    )
    model, checkpoint_report = build_and_load_model(args, device)

    donor_bank, donor_seen = build_donor_bank(
        model,
        loader,
        device,
        args.donors_per_label_direction,
        args.seed,
    )
    records, skip_report = evaluate_interventions(
        model,
        loader,
        donor_bank,
        device,
        args.targets_per_flow,
        args.max_cases,
        not args.allow_same_group,
        not args.no_token_count_matching,
        checkpoint_report["classifier_loaded"],
        args.seed,
    )
    summary = summarize_records(
        records,
        len(label_map),
        args.confidence,
        args.bootstrap_samples,
        args.seed,
    )
    summary.update(
        {
            "metric": "Semantic Intervention Gap",
            "definition": (
                "mean(different-semantic cosine distance - "
                "same-semantic cosine distance)"
            ),
            "label_map": label_map,
            "id_to_label": inverse_label_map(label_map),
            "direction_matched": True,
            "token_count_matched": not args.no_token_count_matching,
            "exclude_same_group": not args.allow_same_group,
            "donor_bank_seen": {
                f"label_{key[0]}_direction_{key[1]}": value
                for key, value in donor_seen.items()
            },
            "skips": skip_report,
            "checkpoint": checkpoint_report,
            "args": vars(args),
        }
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(os.path.join(args.output_dir, "semantic_intervention_gap.json"), summary)
    write_records(os.path.join(args.output_dir, "intervention_cases.csv"), records)
    print(json.dumps({
        "case_count": summary["case_count"],
        "mean_same_semantic_distance": summary["mean_same_semantic_distance"],
        "mean_different_semantic_distance": summary["mean_different_semantic_distance"],
        "semantic_intervention_gap": summary["semantic_intervention_gap"],
        "semantic_intervention_gap_ci": summary["semantic_intervention_gap_ci"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

