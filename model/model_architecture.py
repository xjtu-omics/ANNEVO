import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    def __init__(self, channels, dim_feedforward, num_heads, num_layers, window_size, flank_length, num_blocks, num_base_concat):
        super(FeatureExtractor, self).__init__()
        self.channels = channels
        self.dim_feedforward = dim_feedforward
        self.window_size = window_size
        self.flank_length = flank_length

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=channels, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)

        self.conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = ConvBlock(in_channels=channels * (i + 1), out_channels=channels * (i + 2), kernel_size=3, padding=1)
            self.conv_blocks.append(block)
        # for i in range(num_blocks):
        #     block = ConvBlock(in_channels=channels * (2 ** i), out_channels=channels * (2**(i + 1)), kernel_size=3, padding=1)
        #     self.conv_blocks.append(block)

        self.PositionalEncodingLayer = PositionalEncodingSinCos(d_model=channels * (num_blocks + 1), max_len=int((window_size + 2 * flank_length) / num_base_concat))
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=channels * (num_blocks + 1), nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=num_layers)

        # kaiming_uniform_
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # The shape of x is [batch_size, window_size, num_classes]

        x = x.permute(0, 2, 1)  # Shape of [batch_size, num_classes, window_size]
        x = self.conv1(x)
        residual_1 = x
        x = self.conv2(x)
        x = residual_1 + x

        for block in self.conv_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # Shape of [batch_size, window_size / num_base_concat, channels]
        x = self.PositionalEncodingLayer(x)
        x = self.transformer_encoder(x)  # Shape of [batch_size, window_size / num_base_concat, channels]
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
        return topk_vals, topk_indices


class MoELayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_branches, top_k):
        super(MoELayer, self).__init__()
        self.num_experts = num_branches
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.top_k = top_k

        self.experts = nn.ModuleList([
            SubCladeNet(d_model, dim_feedforward) for _ in range(num_branches)
        ])
        self.gate = TopKGate(d_model, num_branches, top_k)

    def forward(self, x):
        # 输入x形状: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch_size*seq_len, d_model]
        topk_vals, topk_indices = self.gate(x_flat)
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

        return output.view(batch_size, seq_len, d_model), topk_indices, topk_vals


# class TaskLayer(nn.Module):
#     def __init__(self, input_channels, num_classes, num_base_concat):
#         super(TaskLayer, self).__init__()
#         self.num_classes = num_classes
#         self.num_base_concat = num_base_concat
#
#         self.fc = nn.Linear(input_channels, self.num_base_concat * num_classes)
#         nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='leaky_relu')
#
#     def forward(self, x):
#         x = self.fc(x)
#         return x


class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(TransConvBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        torch.nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.trans_conv(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        residual = x
        x = self.conv(x)
        x = x + residual
        return x


class TaskLayer(nn.Module):
    def __init__(self, input_channels, num_classes, num_base_concat, num_blocks):
        super(TaskLayer, self).__init__()
        self.num_classes = num_classes

        self.trans_conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = TransConvBlock(in_channels=input_channels * (num_blocks + 1 - i), out_channels=input_channels * (num_blocks - i), kernel_size=3, padding=1)
            self.trans_conv_blocks.append(block)

        self.fc = nn.Linear(input_channels, num_classes)
        nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for block in self.trans_conv_blocks:
            x = block(x)

        x = x.permute(0, 2, 1)
        x = self.fc(x)

        return x


class ANNEVO(nn.Module):
    def __init__(self, channels, dim_feedforward, num_classes, num_heads, num_encoder_layers, window_size, flank_length, num_blocks, num_branches, top_k):
        super(ANNEVO, self).__init__()
        self.channels = channels
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.window_size = window_size
        self.flank_length = flank_length
        self.num_branch = num_branches
        self.num_base_concat = 2 ** num_blocks
        self.d_model = self.channels * (num_blocks + 1)

        self.FE = FeatureExtractor(channels, dim_feedforward, num_heads, num_encoder_layers, window_size, flank_length, num_blocks, self.num_base_concat)
        self.MoE = MoELayer(self.d_model, dim_feedforward, num_branches, top_k)
        self.task_layer = TaskLayer(channels, num_classes, self.num_base_concat, num_blocks)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.FE(x)
        x, topk_indices, topk_vals = self.MoE(x)
        x = self.task_layer(x)
        # x = x.reshape(batch_size, -1, self.num_classes)
        x = x[:, self.flank_length:self.window_size + self.flank_length, :]

        return x, topk_indices, topk_vals

