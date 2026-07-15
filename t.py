import torch

ckpt = torch.load(
    r"C:\HierarchicalBurst-FlowBERT\models\pretrain_model0525\checkpoint_step_8000.pt",
    map_location="cpu"
)

print(ckpt["args"].get("dataset_info", None))
print(ckpt["model"]["flow_position_embedding.weight"].shape)
print(ckpt["model"]["token_position_embedding.weight"].shape)