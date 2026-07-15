import torch
import torch.nn as nn
import torch.nn.functional as F

from uer.models.hierarchical_burst_flow_bert_uer import (
    build_uer_layer_args,
    UERTransformerStack,
)


class HierarchicalBurstFlowClassifier(nn.Module):
    """
    层次化 Burst-Flow 分类模型。

    结构：
        src: [B, N, L]
          ↓
        word embedding + token position embedding
          ↓
        burst-level UER Transformer
          ↓
        每个 burst 取 [CLS] 向量
          ↓
        burst_vectors: [B, N, H]
          ↓
        flow position embedding + direction embedding
          ↓
        flow-level UER Transformer
          ↓
        mask mean pooling
          ↓
        classifier
          ↓
        logits: [B, num_labels]
    """

    def __init__(
        self,
        vocab_size,
        num_labels,
        max_burst_num=24,
        max_burst_len=128,
        hidden_size=256,
        burst_layers=4,
        flow_layers=2,
        heads_num=4,
        feedforward_size=1024,
        dropout=0.1,
        attention_dropout=0.1,
        pad_id=0,
        direction_vocab_size=3,
        use_direction_embedding=True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.max_burst_num = max_burst_num
        self.max_burst_len = max_burst_len
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        self.use_direction_embedding = use_direction_embedding

        uer_args = build_uer_layer_args(
            hidden_size=hidden_size,
            heads_num=heads_num,
            feedforward_size=feedforward_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_act="gelu",
            layernorm_positioning="pre",
        )

        # 名字尽量和预训练模型保持一致，方便加载 pretrained_model.bin
        self.word_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=pad_id,
        )

        self.token_position_embedding = nn.Embedding(
            max_burst_len,
            hidden_size,
        )

        self.burst_embedding_norm = nn.LayerNorm(hidden_size)
        self.burst_embedding_dropout = nn.Dropout(dropout)

        self.burst_encoder = UERTransformerStack(
            args=uer_args,
            layers_num=burst_layers,
        )

        self.flow_position_embedding = nn.Embedding(
            max_burst_num,
            hidden_size,
        )

        self.direction_embedding = nn.Embedding(
            direction_vocab_size,
            hidden_size,
        )

        self.flow_embedding_norm = nn.LayerNorm(hidden_size)
        self.flow_embedding_dropout = nn.Dropout(dropout)

        self.flow_encoder = UERTransformerStack(
            args=uer_args,
            layers_num=flow_layers,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def build_burst_embeddings(self, src):
        """
        src: [B, N, L]
        return: [B*N, L, H]
        """
        B, N, L = src.shape

        src_flat = src.reshape(B * N, L)

        word_emb = self.word_embedding(src_flat)

        pos_ids = torch.arange(
            L,
            dtype=torch.long,
            device=src.device,
        ).unsqueeze(0).expand(B * N, L)

        pos_emb = self.token_position_embedding(pos_ids)

        emb = word_emb + pos_emb
        emb = self.burst_embedding_norm(emb)
        emb = self.burst_embedding_dropout(emb)

        return emb

    def encode_bursts(self, src, token_mask):
        """
        src:        [B, N, L]
        token_mask: [B, N, L]

        return:
            burst_vectors: [B, N, H]
        """
        B, N, L = src.shape

        emb = self.build_burst_embeddings(src)

        token_mask_flat = token_mask.reshape(B * N, L)

        burst_hidden_flat = self.burst_encoder(
            hidden=emb,
            valid_mask=token_mask_flat,
        )

        burst_hidden = burst_hidden_flat.reshape(
            B,
            N,
            L,
            self.hidden_size,
        )

        # 每个 burst 第 0 位是 [CLS]
        burst_vectors = burst_hidden[:, :, 0, :]

        return burst_vectors

    def encode_flow(self, burst_vectors, direction, burst_mask):
        """
        burst_vectors: [B, N, H]
        direction:     [B, N]
        burst_mask:    [B, N]
        """
        B, N, H = burst_vectors.shape

        pos_ids = torch.arange(
            N,
            dtype=torch.long,
            device=burst_vectors.device,
        ).unsqueeze(0).expand(B, N)

        flow_input = burst_vectors + self.flow_position_embedding(pos_ids)

        if self.use_direction_embedding:
            flow_input = flow_input + self.direction_embedding(direction)

        flow_input = self.flow_embedding_norm(flow_input)
        flow_input = self.flow_embedding_dropout(flow_input)

        flow_hidden = self.flow_encoder(
            hidden=flow_input,
            valid_mask=burst_mask,
        )

        return flow_hidden

    def masked_mean_pooling(self, flow_hidden, burst_mask):
        """
        flow_hidden: [B, N, H]
        burst_mask:  [B, N]

        return:
            flow_vector: [B, H]
        """
        mask = burst_mask.float().unsqueeze(-1)  # [B, N, 1]

        summed = (flow_hidden * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)

        flow_vector = summed / count

        return flow_vector

    def forward(
        self,
        src,
        token_mask,
        burst_mask,
        direction,
        labels=None,
    ):
        burst_vectors = self.encode_bursts(
            src=src,
            token_mask=token_mask,
        )

        flow_hidden = self.encode_flow(
            burst_vectors=burst_vectors,
            direction=direction,
            burst_mask=burst_mask,
        )

        flow_vector = self.masked_mean_pooling(
            flow_hidden=flow_hidden,
            burst_mask=burst_mask,
        )

        logits = self.classifier(flow_vector)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "flow_vector": flow_vector,
            "flow_hidden": flow_hidden,
        }