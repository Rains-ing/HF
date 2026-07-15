import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


from uer.layers.transformer import TransformerLayer


def build_uer_layer_args(
    hidden_size=256,
    heads_num=4,
    feedforward_size=1024,
    dropout=0.1,
    attention_dropout=0.1,
    hidden_act="gelu",
    layernorm_positioning="pre",
):
    """
    构造 UER TransformerLayer 需要的 args。

    这个版本补齐了你当前 UER TransformerLayer 常见会读取的字段。
    """
    return SimpleNamespace(
        # =====================
        # 基础结构参数
        # =====================
        hidden_size=hidden_size,
        heads_num=heads_num,
        feedforward_size=feedforward_size,
        dropout=dropout,
        attention_dropout=attention_dropout,
        hidden_act=hidden_act,

        # =====================
        # LayerNorm 相关
        # =====================
        # 你当前 UER 的 transformer.py 会读取 args.layernorm
        # normal = 普通 LayerNorm
        # t5 = T5LayerNorm
        layernorm="normal",

        # pre / post
        layernorm_positioning=layernorm_positioning,

        # =====================
        # Attention 相关
        # =====================
        remove_attention_scale=False,
        remove_transformer_bias=False,

        # =====================
        # FFN 相关
        # =====================
        # normal = PositionwiseFeedForward
        # gated = GatedFeedForward
        feed_forward="normal",

        # =====================
        # 相对位置编码相关
        # =====================
        relative_position_embedding=False,
        relative_attention_buckets_num=32,
        max_relative_position=128,

        # =====================
        # 兼容字段
        # =====================
        parameter_sharing=False,
        has_residual_attention=False,
        has_lmtarget_bias=True,
        remove_embedding_layernorm=False,
        remove_embedding_dropout=False,
    )

class UERTransformerStack(nn.Module):
    """
    用 UER 的 TransformerLayer 堆叠成 encoder。

    输入:
        hidden: [B, L, H]
        valid_mask: [B, L], 1=有效位置，0=padding

    输出:
        hidden: [B, L, H]
    """

    def __init__(self, args, layers_num):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(copy.deepcopy(args))
            for _ in range(layers_num)
        ])

    def build_attention_mask(self, valid_mask):
        """
        构造适配 UER MultiHeadedAttention 的 attention mask。

        输入:
            valid_mask: [B, L]
                1 = 有效 token / burst
                0 = padding

        输出:
            attn_mask: [B, 1, L, L]

        UER 的 multi_headed_attn.py 中是:
            scores = scores + mask.type_as(scores)

        所以这里不能返回 0/1 mask，
        而要返回 additive mask:
            0.0      = 可见
            -10000.0 = 不可见
        """
        valid_mask = valid_mask.long()

        # 避免全 padding 行导致 attention 全部不可见
        empty_rows = valid_mask.sum(dim=1) == 0
        if empty_rows.any():
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows, 0] = 1

        B, L = valid_mask.shape

        # key 维度可见性: [B, 1, 1, L]
        key_mask = valid_mask.unsqueeze(1).unsqueeze(2)

        # 扩展到 [B, 1, L, L]
        key_mask = key_mask.expand(B, 1, L, L)

        # additive mask:
        # 有效位置 = 0
        # padding 位置 = -10000
        attn_mask = (1.0 - key_mask.float()) * -10000.0

        return attn_mask

    def forward(self, hidden, valid_mask):
        attn_mask = self.build_attention_mask(valid_mask)

        prev_attn = None

        for layer in self.layers:
            # 不同 UER 版本的 TransformerLayer.forward 返回值可能不同：
            # 1. hidden
            # 2. (hidden, prev_attn_out)
            #
            # 所以这里必须兼容 tuple 返回。
            try:
                out = layer(
                    hidden,
                    attn_mask,
                    None,
                    False,
                    prev_attn,
                )
            except TypeError:
                try:
                    out = layer(
                        hidden,
                        attn_mask,
                        None,
                    )
                except TypeError:
                    out = layer(
                        hidden,
                        attn_mask,
                    )

            if isinstance(out, tuple):
                hidden = out[0]

                if len(out) > 1:
                    prev_attn = out[1]
            else:
                hidden = out

        return hidden

class HierarchicalBurstFlowBertUER(nn.Module):
    """
    使用 UER TransformerLayer 构建的层次化 Burst-Flow BERT。

    预训练 V1:
        1. token-level MBM
        2. burst-level MBM

    SODF 五分类:
        先预留接口，后续再加采样器和 loss。
    """

    def __init__(
        self,
        vocab_size,
        max_burst_num=16,
        max_burst_len=128,
        hidden_size=256,
        burst_layers=4,
        flow_layers=2,
        heads_num=4,
        feedforward_size=1024,
        dropout=0.1,
        attention_dropout=0.1,
        pad_id=0,
        mask_id=4,
        direction_vocab_size=3,
        use_direction_embedding=True,
        sodf_classes=5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_burst_num = max_burst_num
        self.max_burst_len = max_burst_len
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.use_direction_embedding = use_direction_embedding
        self.sodf_classes = sodf_classes

        uer_args = build_uer_layer_args(
            hidden_size=hidden_size,
            heads_num=heads_num,
            feedforward_size=feedforward_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_act="gelu",
            layernorm_positioning="pre",
        )

        # =========================
        # 1. Burst-level embedding
        # =========================
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

        # =========================
        # 2. Burst-level UER encoder
        # =========================
        self.burst_encoder = UERTransformerStack(
            args=uer_args,
            layers_num=burst_layers,
        )

        self.token_mbm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )

        # =========================
        # 3. Flow-level embedding
        # =========================
        self.flow_position_embedding = nn.Embedding(
            max_burst_num,
            hidden_size,
        )

        self.direction_embedding = nn.Embedding(
            direction_vocab_size,
            hidden_size,
        )

        self.mask_burst_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_size) * 0.02
        )

        self.flow_embedding_norm = nn.LayerNorm(hidden_size)
        self.flow_embedding_dropout = nn.Dropout(dropout)

        # =========================
        # 4. Flow-level UER encoder
        # =========================
        self.flow_encoder = UERTransformerStack(
            args=uer_args,
            layers_num=flow_layers,
        )

        self.burst_mbm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        # =========================
        # 5. SODF head, 先预留
        # =========================
        self.sodf_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, sodf_classes),
        )

        # =========================
        # 5. nsp head, 先预留
        # =========================
        self.nsp_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
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

        return:
            emb: [B*N, L, H]
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
            burst_hidden:  [B, N, L, H]
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

        return burst_hidden, burst_vectors

    def build_flow_inputs(
        self,
        burst_vectors,
        direction,
        burst_mbm_mask=None,
    ):
        """
        burst_vectors:  [B, N, H]
        direction:      [B, N]
        burst_mbm_mask: [B, N]
        """
        B, N, H = burst_vectors.shape

        clean_burst_targets = burst_vectors.detach()

        flow_input = burst_vectors.clone()

        if burst_mbm_mask is not None:
            mask_vec = self.mask_burst_embedding.expand(B, N, H)

            flow_input = torch.where(
                burst_mbm_mask.unsqueeze(-1).bool(),
                mask_vec,
                flow_input,
            )

        pos_ids = torch.arange(
            N,
            dtype=torch.long,
            device=burst_vectors.device,
        ).unsqueeze(0).expand(B, N)

        flow_input = flow_input + self.flow_position_embedding(pos_ids)

        if self.use_direction_embedding:
            # direction: 0/1 为真实方向，2 为 padding burst
            flow_input = flow_input + self.direction_embedding(direction)

        flow_input = self.flow_embedding_norm(flow_input)
        flow_input = self.flow_embedding_dropout(flow_input)

        return flow_input, clean_burst_targets

    def encode_flow(
        self,
        burst_vectors,
        direction,
        burst_mask,
        burst_mbm_mask=None,
    ):
        """
        burst_vectors:  [B, N, H]
        direction:      [B, N]
        burst_mask:     [B, N]
        burst_mbm_mask: [B, N]
        """
        flow_input, clean_burst_targets = self.build_flow_inputs(
            burst_vectors=burst_vectors,
            direction=direction,
            burst_mbm_mask=burst_mbm_mask,
        )

        flow_hidden = self.flow_encoder(
            hidden=flow_input,
            valid_mask=burst_mask,
        )

        return flow_hidden, clean_burst_targets

    def forward(
        self,
        src,
        token_mask,
        burst_mask,
        direction,
        token_mbm_labels=None,
        burst_mbm_mask=None,
        token_loss_weight=1.0,
        burst_loss_weight=1.0,
        sodf_pairs=None,
        sodf_labels=None,
        sodf_loss_weight=1.0,
        nsp_pairs=None,
        nsp_labels=None,
        nsp_loss_weight=1.0,
    ):
        """
        src:
            [B, N, L]

        token_mbm_labels:
            [B, N, L], 非 mask 位置为 -100

        burst_mbm_mask:
            [B, N]

        sodf_pairs:
            预留，后续格式可以设计为 [M, 4]:
            batch_i, burst_i, batch_j, burst_j

        sodf_labels:
            [M]
        """

        # =========================
        # 1. Burst-level encoder
        # =========================
        burst_hidden, burst_vectors = self.encode_bursts(
            src=src,
            token_mask=token_mask,
        )

        # =========================
        # 2. token-level MBM
        # =========================
        token_logits = self.token_mbm_head(burst_hidden)

        loss_token_mbm = None

        if token_mbm_labels is not None:
            token_logits_flat = token_logits.reshape(-1, self.vocab_size)
            token_labels_flat = token_mbm_labels.reshape(-1)
            valid_token_mbm = token_labels_flat.ne(-100)

            if valid_token_mbm.any():
                loss_token_mbm = F.cross_entropy(
                    token_logits_flat,
                    token_labels_flat,
                    ignore_index=-100,
                )
            else:
                loss_token_mbm = token_logits_flat.sum() * 0.0

        # =========================
        # 3. Flow-level encoder
        # =========================
        flow_hidden, clean_burst_targets = self.encode_flow(
            burst_vectors=burst_vectors,
            direction=direction,
            burst_mask=burst_mask,
            burst_mbm_mask=burst_mbm_mask,
        )

        # =========================
        # 4. burst-level MBM
        # =========================
        burst_pred = self.burst_mbm_head(flow_hidden)

        loss_burst_mbm = None

        if burst_mbm_mask is not None:
            valid_burst_mbm = burst_mbm_mask.bool() & burst_mask.bool()

            # 未归一化
            # if valid_burst_mbm.sum() > 0:
            #     pred = burst_pred[valid_burst_mbm]
            #     target = clean_burst_targets[valid_burst_mbm]

            #     loss_burst_mbm = F.mse_loss(pred, target)
            if valid_burst_mbm.sum() > 0:
                pred = burst_pred[valid_burst_mbm]
                target = clean_burst_targets[valid_burst_mbm]

                # 对每个 burst 向量单独归一化后再算 MSE，避免 hidden 表示范数变大时
                # burst-level MBM loss 被向量尺度平方放大。
                pred = F.layer_norm(pred, pred.shape[-1:])
                target = F.layer_norm(target, target.shape[-1:])

                loss_burst_mbm = F.mse_loss(pred, target)
            else:
                loss_burst_mbm = torch.tensor(
                    0.0,
                    dtype=flow_hidden.dtype,
                    device=flow_hidden.device,
                )

        # =========================
        # 5. SODF, 先预留接口
        # =========================
        loss_sodf = None
        sodf_logits = None

        if sodf_pairs is not None and sodf_labels is not None:
            # sodf_pairs: [M, 4] = b1, n1, b2, n2
            b1 = sodf_pairs[:, 0]
            n1 = sodf_pairs[:, 1]
            b2 = sodf_pairs[:, 2]
            n2 = sodf_pairs[:, 3]

            h1 = flow_hidden[b1, n1]
            h2 = flow_hidden[b2, n2]

            pair_repr = torch.cat([h1, h2], dim=-1)

            sodf_logits = self.sodf_head(pair_repr)

            loss_sodf = F.cross_entropy(
                sodf_logits,
                sodf_labels,
            )

        # =========================
        # 5. NSP, 先预留接口
        # =========================
        loss_nsp = None
        nsp_logits = None

        if nsp_pairs is not None and nsp_labels is not None:
            # nsp_pairs: [M, 4] = b1, n1, b2, n2
            b1 = nsp_pairs[:, 0]
            n1 = nsp_pairs[:, 1]
            b2 = nsp_pairs[:, 2]
            n2 = nsp_pairs[:, 3]

            h1 = flow_hidden[b1, n1]
            h2 = flow_hidden[b2, n2]

            pair_repr = torch.cat([h1, h2], dim=-1)
            nsp_logits = self.nsp_head(pair_repr)

            loss_nsp = F.cross_entropy(
                nsp_logits,
                nsp_labels,
            )

        # =========================
        # 6. total loss
        # =========================
        losses = []

        if loss_token_mbm is not None:
            losses.append(token_loss_weight * loss_token_mbm)

        if loss_burst_mbm is not None:
            losses.append(burst_loss_weight * loss_burst_mbm)

        if loss_sodf is not None:
            losses.append(sodf_loss_weight * loss_sodf)

        if loss_nsp is not None:
            losses.append(nsp_loss_weight * loss_nsp)

        total_loss = sum(losses) if losses else None

        return {
            "loss": total_loss,
            "loss_token_mbm": loss_token_mbm,
            "loss_burst_mbm": loss_burst_mbm,
            "loss_sodf": loss_sodf,
            "token_logits": token_logits,
            "burst_vectors": burst_vectors,
            "flow_hidden": flow_hidden,
            "burst_pred": burst_pred,
            "sodf_logits": sodf_logits,
            "loss_nsp": loss_nsp,
            "nsp_logits": nsp_logits,
        }
