import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from typing import List


class PositionalEncodingSinCos(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncodingSinCos, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}")
        self.dim = dim
        self.base = base

    def _get_inv_freq(self, seq_len: int, device):
        freq_idx = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
        return 1.0 / (self.base ** (freq_idx / self.dim))

    def _get_cos_sin(self, seq_len: int, device, dtype):
        inv_freq = self._get_inv_freq(seq_len, device)
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [L, D/2]
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        return cos, sin

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H, D]
        seq_len = x.size(1)
        cos, sin = self._get_cos_sin(seq_len, x.device, x.dtype)  # [L, D/2]
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        head_dim = d_model // nhead
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")

        self.nhead = nhead
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionEmbedding(head_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_dropout_p = dropout
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B, L, H, D]
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.nhead, self.head_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [B, L, C]
        q = self._reshape_heads(self.q_proj(src))
        k = self._reshape_heads(self.k_proj(src))
        v = self._reshape_heads(self.v_proj(src))

        q = self.rope.apply(q)
        k = self.rope.apply(k)

        # [B, L, H, D] -> [B, H, L, D]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Use PyTorch SDPA so CUDA can dispatch to flash attention kernels when available.
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )  # [B, H, L, D]

        # [B, H, L, D] -> [B, L, C]
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(src.size(0), src.size(1), -1)
        src = self.norm1(src + self.dropout_attn(self.out_proj(attn_out)))

        ffn_out = self.linear2(self.ffn_dropout(F.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout_ffn(ffn_out))
        return src


class RoPETransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RoPETransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv1(x)
        residual = x
        x = self.conv2(x)
        x = x + residual
        x = self.pool1(x)

        return x


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        chs,
        channels,
        dim_feedforward,
        num_heads,
        num_layers,
        window_size,
        flank_length,
        local_pattern_size,
        num_blocks,
    ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=channels, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)

        self.conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = ConvBlock(in_channels=chs[i], out_channels=chs[i+1], kernel_size=3, padding=1)
            self.conv_blocks.append(block)

        self.transformer_encoder = RoPETransformerEncoder(
            d_model=chs[-1],
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=0.1,
        )
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # The shape of x is # (B, L, 4)
        x = x.permute(0, 2, 1)  # [B, 4, L]
        x = self.conv1(x)  # [B, C, L]
        residual_1 = x
        x = self.conv2(x)  # [B, C, L]
        x = residual_1 + x  # [B, C, L]

        for block in self.conv_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # # [B, L/32, C]
        x = self.transformer_encoder(x)
        return x


class SubCladeNet(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(SubCladeNet, self).__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(0.1)
        nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.linear1(x)
        # x = F.gelu(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TopKGate(nn.Module):
    def __init__(self, d_model, num_branches, k):
        super(TopKGate, self).__init__()
        self.k = k
        self.num_experts = num_branches
        self.w_gate = nn.Linear(d_model, num_branches)
        nn.init.kaiming_uniform_(self.w_gate.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: [tokens, d_model]
        logits = self.w_gate(x)  # [tokens, num_experts]
        scores = F.softmax(logits, dim=-1)  # [tokens, num_experts]
        topk_vals, topk_indices = scores.topk(self.k, dim=-1)  # [tokens, k]
        return scores, topk_vals, topk_indices


class MoELayer(nn.Module):
    def __init__(self, chs, dim_feedforward, num_branches, top_k):
        super(MoELayer, self).__init__()
        self.num_experts = num_branches
        self.top_k = top_k
        d_model = chs[-1]

        self.experts = nn.ModuleList([
            SubCladeNet(d_model, dim_feedforward) for _ in range(num_branches)
        ])
        self.gate = TopKGate(d_model, num_branches, top_k)

    def top2_balance_loss(self, top2_indices, top2_vals, scores):
        tokens = top2_indices.shape[0]
        device = top2_indices.device
        num_experts = self.num_experts

        q_counts = torch.zeros(num_experts, device=device, dtype=scores.dtype)
        q_counts.scatter_add_(0, top2_indices.view(-1), torch.ones_like(top2_vals, dtype=scores.dtype).view(-1))
        p_sum_local = scores.sum(dim=0)
        token_count = torch.tensor(float(tokens), device=device, dtype=scores.dtype)

        use_ddp = dist.is_available() and dist.is_initialized()
        if use_ddp:
            # Q has no gradient path (depends on topk indices), aggregate directly.
            dist.all_reduce(q_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

            # Keep local gradient for P while using global merged value in forward.
            p_sum_global_detached = p_sum_local.detach().clone()
            dist.all_reduce(p_sum_global_detached, op=dist.ReduceOp.SUM)
            p_other = p_sum_global_detached - p_sum_local.detach()
            p_sum = p_sum_local + p_other
        else:
            p_sum = p_sum_local

        denom_tokens = token_count.clamp_min(1.0)
        Q = q_counts / (denom_tokens * self.top_k)
        P = p_sum / denom_tokens
        balance_loss = num_experts * torch.sum(P * Q)
        # print(f'Sum(Q): {Q.sum().item():.4f}, Sum(P): {P.sum().item():.4f}, Loss: {balance_loss.item():.4f}')
        return balance_loss

    def forward(self, x):
        # 输入x形状: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch_size*seq_len, d_model]
        scores, topk_vals, topk_indices = self.gate(x_flat)
        output = torch.zeros_like(x_flat)

        for i in range(self.top_k):
            expert_id = topk_indices[:, i]  # [tokens]

            for expert_idx in range(self.num_experts):
                token_mask = (expert_id == expert_idx)
                if token_mask.sum() == 0:
                    continue
                expert_input = x_flat[token_mask]
                expert_output = self.experts[expert_idx](expert_input)
                output[token_mask] += expert_output * topk_vals[token_mask, i].unsqueeze(-1)

        balance_loss = self.top2_balance_loss(topk_indices, topk_vals, scores)
        return output.view(batch_size, seq_len, d_model), balance_loss


class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(TransConvBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        torch.nn.init.kaiming_uniform_(self.trans_conv.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.trans_conv(x)
        x = self.bn1(x)
        # x = F.gelu(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        residual = x
        x = self.conv(x)
        x = x + residual
        return x


class ResolutionRestore(nn.Module):
    def __init__(self, chs, num_classes, num_blocks):
        super(ResolutionRestore, self).__init__()
        self.num_classes = num_classes

        self.trans_conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = TransConvBlock(in_channels=chs[num_blocks-i], out_channels=chs[num_blocks-i-1], kernel_size=3, padding=1)
            self.trans_conv_blocks.append(block)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for block in self.trans_conv_blocks:
            x = block(x)

        return x


class ANNEVO(nn.Module):
    def __init__(self, channels, num_classes, num_heads, window_size, flank_length,
                 num_encoder_layers, n_experts, local_pattern_size, bal_loss_coef):
        super(ANNEVO, self).__init__()

        self.window_size = window_size
        self.flank_length = flank_length
        self.bal_loss_coef = bal_loss_coef
        if local_pattern_size <= 0 or (local_pattern_size & (local_pattern_size - 1)) != 0:
            raise ValueError("local_pattern_size must be a power of 2")
        num_blocks = int(math.log2(local_pattern_size))
        chs_init: List[int] = [
            channels * 1,
            channels * 2,
            channels * 3,
            channels * 4,
            channels * 5,
            channels * 6,
            channels * 8,
        ]
        chs = chs_init[0:num_blocks+1]
        dim_feedforward = 2 * chs[-1]
        self.FE = FeatureExtractor(chs, channels, dim_feedforward, num_heads, num_encoder_layers, window_size,
                                   flank_length, local_pattern_size, num_blocks=num_blocks)
        self.MoE = MoELayer(chs, dim_feedforward, n_experts, top_k=2)
        self.resolution_restore = ResolutionRestore(chs, num_classes, num_blocks=num_blocks)
        self.classifier = nn.Linear(chs[0], num_classes)
        nn.init.kaiming_uniform_(self.classifier.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, seq):
        dna = seq
        dna = self.FE(dna)
        dna, balance_loss = self.MoE(dna)
        dna = self.resolution_restore(dna)
        x = dna
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        x = x[:, self.flank_length:self.window_size + self.flank_length, :]
        return x, balance_loss * self.bal_loss_coef
