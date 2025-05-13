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
        self.ln1 = nn.LayerNorm([2000])
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
    def __init__(self, channels, dim_feedforward, nheads, num_layers, window_size, flank_length, num_blocks):
        super(FeatureExtractor, self).__init__()
        self.num_base_concat = 2 ** (num_blocks + 1)
        self.channels = channels
        self.dim_feedforward = dim_feedforward
        self.window_size = window_size
        self.flank_length = flank_length

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=channels, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = ConvBlock(in_channels=channels * (i + 1), out_channels=channels * (i + 2), kernel_size=3, padding=1)
            self.conv_blocks.append(block)

        self.PositionalEncodingLayer = PositionalEncodingSinCos(d_model=channels * (num_blocks + 1), max_len=int((window_size + 2 * flank_length) / self.num_base_concat))
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=channels * (num_blocks + 1), nhead=nheads, dim_feedforward=dim_feedforward, batch_first=True)
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
        x = self.pool1(x)

        for block in self.conv_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # Shape of [batch_size, window_size / num_base_concat, channels]
        x = self.PositionalEncodingLayer(x)
        x = self.transformer_encoder(x)  # Shape of [batch_size, window_size / num_base_concat, channels]
        return x


class EvoBranch(nn.Module):
    def __init__(self, channels, dim_feedforward, nheads, num_layers, window_size, flank_length, num_blocks):
        super(EvoBranch, self).__init__()
        self.num_base_concat = 2 ** (num_blocks + 1)
        self.channels = channels
        self.dim_feedforward = dim_feedforward
        self.window_size = window_size
        self.flank_length = flank_length

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=channels * (num_blocks + 1), nhead=nheads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)  # Shape of [batch_size, window_size / num_base_concat, channels]
        return x


class Experts(nn.Module):
    def __init__(self, channels, dim_feedforward, nheads, num_layers, window_size, flank_length, num_blocks):
        super(Experts, self).__init__()
        self.num_base_concat = 2 ** (num_blocks + 1)
        self.channels = channels
        self.dim_feedforward = dim_feedforward
        self.window_size = window_size
        self.flank_length = flank_length

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=channels * (num_blocks + 1), nhead=nheads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)  # Shape of [batch_size, window_size / num_base_concat, channels]
        return x


class TaskTower(nn.Module):
    """Tower"""

    def __init__(self, window_size, flank_length, channels, num_classes, num_blocks):
        super(TaskTower, self).__init__()
        self.num_base_concat = 2 ** (num_blocks + 1)
        self.num_classes = num_classes
        self.window_size = window_size
        self.flank_length = flank_length
        self.fc = nn.Linear(channels * (num_blocks + 1), self.num_base_concat * num_classes)
        # kaiming_uniform_
        nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.fc(x)  # Shape of [batch_size, window_size / num_base_concat, num_classes * num_base_concat]
        if self.num_base_concat > 1:
            x = x.view(-1, x.shape[1] * self.num_base_concat, self.num_classes)  # Shape of [batch_size, window_size, num_classes]
        x = x[:, self.flank_length:self.window_size + self.flank_length, :]
        return x


class GateNetwork(nn.Module):
    """Gate network to dynamically weigh expert contributions."""

    def __init__(self, channels, dim_feedforward, window_size, flank_length, num_blocks, weights_shape):
        super(GateNetwork, self).__init__()
        self.num_base_concat = 2 ** (num_blocks + 1)
        self.channels = channels
        self.dim_feedforward = dim_feedforward
        self.window_size = window_size
        self.flank_length = flank_length

        self.weights = nn.Parameter(torch.randn(int((window_size + 2 * flank_length) / self.num_base_concat)))
        self.fc = nn.Linear(channels * (num_blocks + 1), weights_shape)
        nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # The shape of x is [batch_size, window_size, num_classes]

        normalized_weights = F.softmax(self.weights, dim=0)
        weighted_outputs = x * normalized_weights.view(1, -1, 1)
        x = weighted_outputs.sum(dim=1)
        x = self.fc(x)
        weights = F.softmax(x, dim=1)
        return weights


class EvoNetwork(nn.Module):
    def __init__(self, channels, dim_feedforward, num_classes_base, num_classes_transition, num_classes_phases, nheads, num_layers, window_size, flank_length, num_blocks, num_branches):
        super(EvoNetwork, self).__init__()
        self.num_base_concat = 2 ** (num_blocks + 1)
        self.channels = channels
        self.dim_feedforward = dim_feedforward
        self.num_classes_base = num_classes_base
        self.num_classes_phases = num_classes_phases
        self.num_classes_transition = num_classes_transition
        self.window_size = window_size
        self.flank_length = flank_length
        self.num_branch = num_branches

        self.FE = FeatureExtractor(channels, dim_feedforward, nheads, 1, window_size, flank_length, num_blocks)
        self.gate_branch = GateNetwork(channels, dim_feedforward, window_size, flank_length, num_blocks, num_branches)
        self.task_layers1 = TaskTower(window_size, flank_length, channels, num_classes_base, num_blocks)
        self.task_layers2 = TaskTower(window_size, flank_length, channels, num_classes_transition, num_blocks)
        self.task_layers3 = TaskTower(window_size, flank_length, channels, num_classes_phases, num_blocks)
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            branch = EvoBranch(channels, dim_feedforward, nheads, num_layers, window_size, flank_length, num_blocks)
            self.branches.append(branch)

        self.experts_share = Experts(channels, dim_feedforward, nheads, num_layers, window_size, flank_length, num_blocks)
        self.experts_base = Experts(channels, dim_feedforward, nheads, num_layers, window_size, flank_length, num_blocks)
        self.experts_transition = Experts(channels, dim_feedforward, nheads, num_layers, window_size, flank_length, num_blocks)
        self.experts_phase = Experts(channels, dim_feedforward, nheads, num_layers, window_size, flank_length, num_blocks)
        self.gate_base = GateNetwork(channels, dim_feedforward, window_size, flank_length, num_blocks, 2)
        self.gate_transition = GateNetwork(channels, dim_feedforward, window_size, flank_length, num_blocks, 2)
        self.gate_phase = GateNetwork(channels, dim_feedforward, window_size, flank_length, num_blocks, 2)

    def forward(self, x):
        x = self.FE(x)
        branch_outputs = []
        for i in range(self.num_branch):
            output = self.branches[i](x)
            branch_outputs.append(output)
        branch_outputs = torch.stack(branch_outputs, dim=1)
        gate_weights_branches = self.gate_branch(x).unsqueeze(-1).unsqueeze(-1)
        weighted_branches_out = torch.sum(branch_outputs * gate_weights_branches, dim=1)

        expert_share_out = self.experts_share(weighted_branches_out)
        expert_base_out = self.experts_base(weighted_branches_out)
        expert_transition_out = self.experts_transition(weighted_branches_out)
        expert_phase_out = self.experts_transition(weighted_branches_out)

        base_outputs = [expert_share_out, expert_base_out]
        base_outputs = torch.stack(base_outputs, dim=1)
        gate_weights_base = self.gate_base(x).unsqueeze(-1).unsqueeze(-1)
        weighted_base_out = torch.sum(base_outputs * gate_weights_base, dim=1)

        transition_outputs = [expert_share_out, expert_transition_out]
        transition_outputs = torch.stack(transition_outputs, dim=1)
        gate_weights_transition = self.gate_transition(x).unsqueeze(-1).unsqueeze(-1)
        weighted_transition_out = torch.sum(transition_outputs * gate_weights_transition, dim=1)

        phase_outputs = [expert_share_out, expert_phase_out]
        phase_outputs = torch.stack(phase_outputs, dim=1)
        gate_weights_phase = self.gate_phase(x).unsqueeze(-1).unsqueeze(-1)
        weighted_phase_out = torch.sum(phase_outputs * gate_weights_phase, dim=1)

        x1 = self.task_layers1(weighted_base_out)
        x2 = self.task_layers2(weighted_transition_out)
        x3 = self.task_layers3(weighted_phase_out)
        return x1, x2, x3
