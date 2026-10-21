# Hierarchical Flow Transformer (HF)

PyTorch source code for **Hierarchical Flow Transformer: Encrypted Traffic Detection via Nested Attention and Cross-Granularity Pretraining**, by Yongcai Xiao and Yuping Lai.

Repository: [Rains-ing/HF](https://github.com/Rains-ing/HF)

## Overview

HF learns representations of encrypted network traffic using a token-burst-flow hierarchy. A burst contains consecutive packets traveling in the same direction within a bidirectional flow.

The model first encodes tokens within each burst, then models relationships between bursts using their position and transmission direction. Self-supervised pretraining combines token-level masked prediction, full-burst representation reconstruction, and burst relation learning. The hierarchical backbone is subsequently fine-tuned for traffic classification.

```text
PCAP / PCAPNG
    -> bidirectional flows
    -> same-direction bursts
    -> token sequences
    -> burst-level Transformer
    -> flow-level Transformer
    -> masked mean pooling
    -> traffic classifier
```

## Repository structure

| Path | Purpose |
| --- | --- |
| `pipeline.py` | Unified preprocessing, pretraining, and fine-tuning entry point |
| `process_pretrain_data/` | Flow/session construction, tokenization, vocabulary building, and pretraining data preparation |
| `process_fine_tuning/` | Labeled traffic preprocessing and dataset preparation |
| `pre-train/` | Single-process and distributed pretraining, including gradient accumulation |
| `fine-tuning/finetune_hierarchical.py` | Supervised fine-tuning and validation/test evaluation |
| `inference/test_finetuned_hierarchical.py` | Standalone evaluation of a fine-tuned model |
| `uer/models/hierarchical_burst_flow_bert_uer.py` | Hierarchical pretraining model |
| `uer/models/hierarchical_burst_flow_classifier.py` | Downstream classifier |
| `experiments/burst_semantics/` | Additional representation-analysis utilities |
| `tests/` | Tests for session construction and the fine-tuning learning-rate schedule |

## Installation

Use Python 3.10 or later in a dedicated environment.

```bash
git clone https://github.com/Rains-ing/HF.git
cd HF
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the dependencies used by the main workflow:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib scapy six regex tokenizers tqdm PyYAML
```

For GPU training, select a PyTorch build compatible with your CUDA environment using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The repository's `requirements.txt` records a pinned dependency snapshot, including CUDA-specific PyTorch packages; it is not a platform-independent lock file. The manuscript reports Python 3.10.20 and PyTorch 2.5.1, while the dependency snapshot records a different PyTorch version.

Run commands from the repository root. The examples below use Bash line continuation; in PowerShell, place each command on one line or use PowerShell continuation syntax.

## Data preparation

Obtain the datasets from their original providers and follow their applicable access and usage conditions. The manuscript uses USTC-TFC2016 and CICAndMal2017 for pretraining, and CSTNET-TLS 1.3, ISCX-VPN-App, ISCX-VPN-Service, and ITC-Net-Blend-60 for downstream evaluation.

The pipeline accepts a PCAP/PCAPNG file or a directory of capture files. With `--label_mode parent`, the immediate parent directory of each capture supplies its class label:

```text
data/
  pretrain/
    capture_001.pcap
    capture_002.pcap
  downstream/
    application_a/
      capture_001.pcap
    application_b/
      capture_002.pcap
```

Preprocessing groups packets into bidirectional sessions and same-direction bursts, converts packet bytes to tokens, and writes sharded PyTorch data. Fine-tuning preprocessing replaces source/destination IP addresses and ports with fixed values. The default hexadecimal tokenizer uses two bytes per token.

The unified fine-tuning pipeline currently performs a label-stratified split of flow samples. This differs from the PCAP-file-level split described in the manuscript. For a PCAP-file-level evaluation, partition captures before preprocessing and use the standalone fine-tuning script with separately prepared training, validation, and test files. Preserve split membership and label IDs across all methods being compared.

## Pretraining

The following example uses the manuscript's encoder dimensions and explicitly selects 16 bursts, 256 token positions per burst, and batch size 32.

```bash
python pipeline.py pretrain \
  --pcap_input data/pretrain \
  --output_root runs/pretrain \
  --run_name hf_pretrain \
  --tokenizer hex --token_bytes 2 \
  --max_burst_num 16 --max_burst_len 256 --min_burst_num 2 \
  --hidden_size 768 --burst_layers 8 --flow_layers 4 \
  --num_heads 12 --intermediate_size 3072 \
  --batch_size 32 --lr 2e-5 --max_steps 100000 \
  --relation_task sodf --seed 42 \
  --pcap_num_workers 4 --jsonl_to_pt_workers 4
```

The pipeline writes a run configuration, a parameter guide, prepared data, training outputs, and a `*_pipeline_outputs.json` file containing the generated paths. Reuse its `pretrained_model_path`, `vocab_path`, and `dataset_info_path` values for fine-tuning.

For distributed pretraining, add `--cuda_devices 0,1 --nproc_per_node 2`. Effective global batch size is `batch_size * nproc_per_node * gradient_accumulation_steps`.

These examples demonstrate the current workflow rather than a verified reproduction of every reported experiment. Some CLI defaults differ from the manuscript. In particular, loss weights and masking probabilities are set by `PretrainOptions` in `pipeline.py`; they can also be configured through the standalone pretraining script. Keep the exact settings and data partitions with each experiment.

## Fine-tuning

Replace the three asset paths below with values from the pretraining output manifest.

```bash
python pipeline.py finetune \
  --pcap_input data/downstream \
  --output_root runs/finetune \
  --run_name hf_finetune \
  --pretrained_model_path /path/to/pretrained_model.bin \
  --vocab_path /path/to/hex_vocab.txt \
  --dataset_info_path /path/to/dataset_info.json \
  --label_mode parent \
  --hidden_size 768 --burst_layers 8 --flow_layers 4 \
  --num_heads 12 --intermediate_size 3072 \
  --batch_size 32 --epochs 10 --lr 1e-4 \
  --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1 \
  --seed 42 --pcap_num_workers 4 --jsonl_to_pt_workers 4
```

The pipeline reuses pretraining tokenization metadata and sequence dimensions. Encoder dimensions must match the supplied model. Fine-tuning includes validation and test evaluation and records the output paths in `*_pipeline_outputs.json`.

## Evaluation

Use the generated `test_path` and `finetuned_model_path` from the fine-tuning output manifest. Supply the same pretraining `dataset_info.json` and encoder dimensions used for training.

```bash
python inference/test_finetuned_hierarchical.py \
  --test_path /path/to/test.pt \
  --dataset_info /path/to/dataset_info.json \
  --model_path /path/to/finetuned_model.bin \
  --output_dir runs/evaluation \
  --hidden_size 768 --burst_layers 8 --flow_layers 4 \
  --num_heads 12 --intermediate_size 3072 \
  --batch_size 32
```

Evaluation reports accuracy, macro-precision, macro-recall, macro-F1, per-class results, and a confusion matrix. It saves `test_metrics.json` and `test_predictions.tsv`, together with text reports. Ensure the test data preserve the label mapping used during training and include the expected class IDs.

## Command reference

```bash
python pipeline.py pretrain --help
python pipeline.py finetune --help
python pre-train/pretrain_hierarchical_ddp_grad_accum.py --help
python fine-tuning/finetune_hierarchical.py --help
python inference/test_finetuned_hierarchical.py --help
```

## Acknowledgments and license

This project uses code based on [UER-py](https://github.com/dbiir/UER-py). See [NOTICE](NOTICE) for attribution.

The project source code is distributed under the [Apache License 2.0](LICENSE). Third-party code retains its applicable copyright and attribution notices. External datasets remain subject to their original providers' terms.
