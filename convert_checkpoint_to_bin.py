import os
import argparse
import torch


def extract_model_state_dict(obj):
    """
    从不同格式的 .pt 文件中提取模型权重。

    支持：
    1. 完整 checkpoint:
       {
           "step": ...,
           "model": model.state_dict(),
           "optimizer": ...,
           "args": ...
       }

    2. 常见格式:
       {
           "state_dict": ...
       }

    3. 纯 state_dict:
       {
           "word_embedding.weight": tensor(...),
           ...
       }
    """

    if not isinstance(obj, dict):
        raise ValueError(
            "输入 .pt 不是 dict 类型。"
            "如果这是 hierarchical_dataset.pt，它是数据集文件，不能转换成模型权重。"
        )

    # 情况 1：我们当前保存的 checkpoint 格式
    if "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]

    # 情况 2：其他常见 checkpoint 格式
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]

    # 情况 3：本身就是 state_dict
    # 判断方式：dict 的 value 大部分是 tensor
    tensor_values = [
        v for v in obj.values()
        if torch.is_tensor(v)
    ]

    if len(tensor_values) > 0:
        return obj

    raise ValueError(
        "没有在 .pt 中找到模型权重。"
        "请确认输入的是 checkpoint_step_xxx.pt 或 checkpoint_final.pt，"
        "而不是数据集 hierarchical_dataset.pt。"
    )


def remove_module_prefix(state_dict):
    """
    如果模型是 DataParallel / DDP 保存的，key 可能带 module. 前缀。
    这里自动去掉。
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        new_state_dict[new_key] = value

    return new_state_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/models/pretrain_model0602/checkpoint_step_36000.pt",
        help="输入 checkpoint .pt，例如 checkpoint_step_10.pt 或 checkpoint_final.pt",
    )

    parser.add_argument(
        "--output_bin",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/models/model_bin/pretrained_model_0602_36000.bin",
        help="输出模型权重 .bin，例如 pretrained_model.bin",
    )

    parser.add_argument(
        "--remove_module_prefix",
        action="store_true",
        default=True,
        help="如果 state_dict 的 key 带 module. 前缀，则去掉",
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    print(f"[LOAD] {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    state_dict = extract_model_state_dict(ckpt)

    if args.remove_module_prefix:
        state_dict = remove_module_prefix(state_dict)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_bin)), exist_ok=True)

    torch.save(state_dict, args.output_bin)

    print(f"[SAVE] {args.output_bin}")
    print(f"[INFO] num parameters tensors: {len(state_dict)}")

    # 打印前几个 key，方便确认
    print("\n[STATE_DICT KEYS SAMPLE]")
    for i, key in enumerate(state_dict.keys()):
        print(key)
        if i >= 20:
            break


if __name__ == "__main__":
    main()