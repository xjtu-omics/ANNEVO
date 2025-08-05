import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B*L, C] - raw logits
        targets: [B*L]   - class indices
        """
        log_probs = F.log_softmax(inputs, dim=-1)  # [B*L, C]
        probs = log_probs.exp()  # [B*L, C]

        # Gather log-probabilities and probabilities of the true class
        pt = probs[torch.arange(len(targets)), targets]  # [B*L]
        log_pt = log_probs[torch.arange(len(targets)), targets]  # [B*L]

        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * log_pt  # [B*L]

        if self.reduction == 'none':
            return loss  # [B*L]
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, positive_classes=None):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.positive_classes = positive_classes

    def forward(self, y_pred, y_true):
        # [B, L] → [B, L, 1] → binary mask
        B, L, C = y_pred.shape
        device = y_pred.device
        y_true_binary = torch.isin(y_true, torch.tensor(self.positive_classes, device=device)).float()  # [B, L]
        y_true_binary = y_true_binary.unsqueeze(-1)

        y_pred_prob = F.softmax(y_pred, dim=-1)  # [B, L, C]
        y_pred_pos_prob = y_pred_prob[..., self.positive_classes].sum(dim=-1, keepdim=True)  # [B, L, 1]

        intersection = torch.sum(y_pred_pos_prob * y_true_binary, dim=(1, 2))  # [B]
        union = torch.sum(y_pred_pos_prob + y_true_binary, dim=(1, 2))  # [B]
        has_pos_gt = torch.sum(y_true_binary, dim=(1, 2)) > self.epsilon  # [B]
        dice_loss = 1 - (2 * intersection + self.epsilon) / (union + self.epsilon)  # [B]

        y_pred_classes = torch.argmax(y_pred, dim=-1)  # [B, L]
        pred_pos_mask = torch.isin(y_pred_classes, torch.tensor(self.positive_classes, device=device)).float()  # [B, L]

        pred_pos_mask = pred_pos_mask.unsqueeze(-1)  # [B, L, 1]
        pred_pos_scores = y_pred_pos_prob  # [B, L, 1]
        fp_total = (pred_pos_scores * pred_pos_mask).sum(dim=(1, 2))  # [B]
        fp_count = pred_pos_mask.sum(dim=(1, 2)).clamp(min=1)  # [B]
        fp_loss = fp_total / fp_count  # [B]

        final_loss = torch.where(has_pos_gt, dice_loss, fp_loss)

        return final_loss


def top2_balance_loss(top2_indices, top2_vals, num_experts):
    """
    GShard-style Load Balancing Loss for Top-2 routing.

    Args:
        top2_indices: [tokens, 2] - expert indices for top-2
        top2_vals: [tokens, 2] - softmax scores for top-2 experts
        num_experts: total number of experts

    Returns:
        Scalar balance loss
    """
    tokens = top2_indices.shape[0]
    device = top2_indices.device

    Q = torch.zeros(num_experts, device=device)
    Q.scatter_add_(0, top2_indices.view(-1), torch.ones_like(top2_vals).view(-1))
    Q /= (tokens * 2)  # 每个 token 分给两个专家

    P = torch.zeros(num_experts, device=device)
    P.scatter_add_(0, top2_indices.view(-1), top2_vals.view(-1))
    P /= (tokens * 2)

    # Loss: E * sum(P_i * Q_i)
    balance_loss = num_experts * torch.sum(P * Q)
    return balance_loss


def multi_loss(outputs, topk_indices, topk_vals, num_branches, labels, position_mask, loss_fn_CE, loss_fn_dice_CDS0, loss_fn_dice_CDS1, loss_fn_dice_CDS2,
               loss_fn_dice_intron, coefficient):
    loss_dice_CDS0 = loss_fn_dice_CDS0(outputs, labels)
    loss_dice_CDS1 = loss_fn_dice_CDS1(outputs, labels)
    loss_dice_CDS2 = loss_fn_dice_CDS2(outputs, labels)
    loss_dice_intron = loss_fn_dice_intron(outputs, labels)
    loss_balance = top2_balance_loss(topk_indices, topk_vals, num_branches)

    position_mask = position_mask.reshape(-1)
    outputs = outputs.reshape(-1, 5)
    labels = labels.reshape(-1)
    loss_CE = loss_fn_CE(outputs, labels) * position_mask

    loss = loss_CE.mean() + 0.01 * loss_balance.mean() + coefficient * (loss_dice_CDS0.mean() + loss_dice_CDS1.mean() + loss_dice_CDS2.mean()) + coefficient * loss_dice_intron.mean()
    return loss, loss_CE, loss_dice_CDS0, loss_dice_CDS1, loss_dice_CDS2, loss_dice_intron, loss_balance
