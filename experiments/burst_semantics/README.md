# Burst Semantic Evaluation

This directory implements two evaluations for the hierarchical Burst–Flow
encoder:

1. **Contextual Semantic Gain (CSG)** compares a frozen linear probe on local
   Burst-Encoder vectors with the same probe on contextual Flow-Encoder burst
   vectors.
2. **Semantic Intervention Gap (SIG)** replaces a target burst representation
   with a same-semantic donor and a different-semantic donor, then compares the
   induced drift of the complete flow representation.

The experiments never train or update the hierarchical encoder.

## Burst semantic annotation format

Existing PT samples can directly contain a `burst_semantic_labels` vector of
length `max_burst_num`, using `-100` for unannotated positions.  Alternatively,
annotations can be kept in a JSONL sidecar so the large PT files do not need to
be rewritten.

One JSON object corresponds to one PT sample:

```json
{"sample_index": 0, "group_id": "capture_day_01", "burst_labels": ["setup", "client_control", "server_bulk", null]}
{"sample_index": 1, "group_id": "capture_day_02", "burst_labels": ["setup", "client_control", "server_control", "teardown"]}
```

Rules:

- Use `sample_index` when the PT file does not retain `flow_id`.
- `flow_id` may be used instead when PT samples were generated with metadata.
- `burst_labels[i]` labels burst position `i` before padding.
- `null`, `unknown`, `ignore`, and `-100` are ignored.
- `group_id` should identify a PCAP, device, user, day, or session family. SIG
  excludes donors from the same group by default.

String labels are mapped to contiguous IDs automatically across all supplied
splits.  For a fixed mapping, pass a JSON file such as:

```json
{
  "setup": 0,
  "client_control": 1,
  "server_control": 2,
  "client_bulk": 3,
  "server_bulk": 4,
  "keepalive": 5,
  "teardown": 6
}
```

Train, validation, and test PT files must be split by PCAP/device/time before
annotation.  Do not randomly split bursts from related flows across sets.

## 1. Contextual Semantic Gain

The experiment extracts two representations for every annotated burst:

```text
local       = BurstEncoder(...)[CLS]
contextual  = FlowEncoder(local burst sequence)[burst position]
```

It standardizes both representations with train-only statistics and trains two
linear probes with identical optimization settings.

```text
CSG = contextual test Macro-F1 - local test Macro-F1
```

Example:

```bash
python experiments/burst_semantics/contextual_semantic_gain.py \
  --train_data /path/to/train.pt \
  --valid_data /path/to/valid.pt \
  --test_data /path/to/test.pt \
  --train_annotations /path/to/train_burst_semantics.jsonl \
  --valid_annotations /path/to/valid_burst_semantics.jsonl \
  --test_annotations /path/to/test_burst_semantics.jsonl \
  --semantic_label_map /path/to/burst_label_map.json \
  --dataset_info /path/to/dataset_info.json \
  --checkpoint /path/to/pretrained_model.bin \
  --hidden_size 256 \
  --burst_layers 4 \
  --flow_layers 2 \
  --num_heads 4 \
  --intermediate_size 1024 \
  --output_dir /path/to/results/csg
```

Outputs:

- `contextual_semantic_gain.json`: metrics, confusion matrices, label mapping,
  checkpoint loading report, and CSG.
- `linear_probes.pt`: best local and contextual linear-probe weights.

Interpretation:

- `local Macro-F1` measures whether Burst Encoder vectors carry recognizable
  local action information.
- Positive `CSG` measures whether the surrounding flow gives each burst a more
  informative contextual role.
- Report per-class results; a positive average can otherwise hide degradation
  on rare semantic roles.

## 2. Semantic Intervention Gap

For every target burst, SIG selects:

- a donor with the same semantic label and direction;
- a donor with a different semantic label but the same direction.

By default, donors are also chosen with the closest available token count and
cannot come from the same `group_id`.  Replacements happen at the local burst
representation, while target flow position and direction are preserved.  This
isolates semantic content better than replacing padded raw tensors of different
lengths.

```text
D_same = 1 - cosine(clean flow, same-semantic replacement flow)
D_diff = 1 - cosine(clean flow, different-semantic replacement flow)
SIG    = mean(D_diff - D_same)
```

Example:

```bash
python experiments/burst_semantics/semantic_intervention_gap.py \
  --data /path/to/test.pt \
  --annotations /path/to/test_burst_semantics.jsonl \
  --semantic_label_map /path/to/burst_label_map.json \
  --dataset_info /path/to/dataset_info.json \
  --checkpoint /path/to/finetuned_model.bin \
  --num_flow_labels 20 \
  --hidden_size 256 \
  --burst_layers 4 \
  --flow_layers 2 \
  --num_heads 4 \
  --intermediate_size 1024 \
  --targets_per_flow 2 \
  --donors_per_label_direction 1000 \
  --bootstrap_samples 5000 \
  --output_dir /path/to/results/sig
```

Outputs:

- `semantic_intervention_gap.json`: SIG, bootstrap confidence interval,
  per-class gaps, donor statistics, and optional flow-prediction flip rates.
- `intervention_cases.csv`: every paired replacement and its distances.

When a fine-tuned classifier head is successfully loaded, the experiment also
reports same-semantic and different-semantic prediction-flip rates.  When a
pretraining checkpoint is used, embedding-level SIG is still computed.

A useful result should satisfy all of the following:

```text
D_diff > D_same
SIG > 0
bootstrap lower confidence bound > 0
different-semantic prediction flip rate > same-semantic flip rate
```

## Required controls

Run both experiments for at least these model variants:

```text
random initialization
supervised training without pretraining
token masking only
whole-burst masking only
token + whole-burst masking
full model with relation learning
full model without direction embedding
full model with shuffled burst order
```

The two semantic experiments should not be treated as sufficient evidence by
themselves.  Also report group-held-out flow classification, label efficiency,
and checks that IP/port/capture identity cannot be decoded from the learned
representations.

## Environment note

Use the same PyTorch environment as pretraining and fine-tuning.  Model
dimensions passed on the command line must match the checkpoint.  The scripts
stop if no Burst Encoder or Flow Encoder tensors can be loaded.

