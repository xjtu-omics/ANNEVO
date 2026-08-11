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


# class DiceLoss(nn.Module):
#     def __init__(self, epsilon=1e-8, positive_classes=None):
#         super(DiceLoss, self).__init__()
#         self.epsilon = epsilon
#         self.positive_classes = positive_classes
#
#     def forward(self, y_pred, y_true):
#         # [B, L] → [B, L, 1] → binary mask
#         B, L, C = y_pred.shape
#         device = y_pred.device
#         y_true_binary = torch.isin(y_true, torch.tensor(self.positive_classes, device=device)).float()  # [B, L]
#         y_true_binary = y_true_binary.unsqueeze(-1)
#
#         y_pred_prob = F.softmax(y_pred, dim=-1)  # [B, L, C]
#         y_pred_pos_prob = y_pred_prob[..., self.positive_classes].sum(dim=-1, keepdim=True)  # [B, L, 1]
#
#         intersection = torch.sum(y_pred_pos_prob * y_true_binary, dim=(1, 2))  # [B]
#         union = torch.sum(y_pred_pos_prob + y_true_binary, dim=(1, 2))  # [B]
#         has_pos_gt = torch.sum(y_true_binary, dim=(1, 2)) > self.epsilon  # [B]
#         dice_loss = 1 - (2 * intersection + self.epsilon) / (union + self.epsilon)  # [B]
#
#         y_pred_classes = torch.argmax(y_pred, dim=-1)  # [B, L]
#         pred_pos_mask = torch.isin(y_pred_classes, torch.tensor(self.positive_classes, device=device)).float()  # [B, L]
#
#         pred_pos_mask = pred_pos_mask.unsqueeze(-1)  # [B, L, 1]
#         pred_pos_scores = y_pred_pos_prob  # [B, L, 1]
#         fp_total = (pred_pos_scores * pred_pos_mask).sum(dim=(1, 2))  # [B]
#         fp_count = pred_pos_mask.sum(dim=(1, 2)).clamp(min=1)  # [B]
#         fp_loss = fp_total / fp_count  # [B]
#
#         final_loss = torch.where(has_pos_gt, dice_loss, fp_loss)
#
#         return final_loss


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, classes_list=None, classes_weights=None, fp_weight=1):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        if classes_list is None:
            raise ValueError("The classes_list field must be provided to define the classes for which loss needs to be calculated.")

        self.classes_list = classes_list
        self.num_classes = len(classes_list)
        self.fp_weight = fp_weight

        if classes_weights is not None:
            if len(classes_weights) != self.num_classes:
                raise ValueError("The length of classes_weights must match that of classes_list.")
            weights_tensor = torch.tensor(classes_weights, dtype=torch.float32)
        else:
            weights_tensor = torch.ones(self.num_classes, dtype=torch.float32)
        self.register_buffer('classes_weights', weights_tensor)

    def single_loss_calculation(self, y_pred_prob, y_true, class_id, y_pred_classes):
        y_pred_c = y_pred_prob[..., class_id]
        y_true_c = (y_true == class_id).float()

        intersection = torch.sum(y_pred_c * y_true_c, dim=1)
        union = torch.sum(y_pred_c + y_true_c, dim=1)  # [B]
        dice_loss = 1.0 - (2.0 * intersection + self.epsilon) / (union + self.epsilon)

        pred_is_current_class = (y_pred_classes == class_id).float()
        fp_mask = pred_is_current_class * (1.0 - y_true_c)
        fp_total_prob = torch.sum(y_pred_c * fp_mask, dim=1)
        fp_count = fp_mask.sum(dim=1).clamp(min=1)
        fp_loss = self.fp_weight * (fp_total_prob / fp_count)

        has_object = torch.sum(y_true_c, dim=1) > 0
        final_loss = torch.where(
            has_object,
            dice_loss,
            fp_loss
        )

        return final_loss.mean()

    def forward(self, y_pred, y_true):
        y_pred_prob = F.softmax(y_pred, dim=-1)  # Shape [B, L, C]
        y_pred_classes = torch.argmax(y_pred_prob, dim=-1)
        total_loss = 0.0

        for i, class_id in enumerate(self.classes_list):
            class_avg_loss = self.single_loss_calculation(y_pred_prob, y_true, class_id, y_pred_classes)
            weight = self.classes_weights[i]
            total_loss += weight * class_avg_loss

        return total_loss

    # def single_loss_calculation(self, y_pred_prob, y_true, class_id):
    #     """
    #     y_pred_prob: [B, L, C] (Softmax probabilities)
    #     y_true: [B, L] (Class indices)
    #     """
    #     y_pred_c = y_pred_prob[..., class_id]  # Shape [B, L]
    #     y_true_c = (y_true == class_id).float()  # Shape [B, L] (Binary mask for this class)
    #
    #     # 2. 计算交集 (Intersection) 和并集 (Union) - 沿序列维度 L 求和
    #     # Intersection = 2 * sum(p*y)
    #     intersection = torch.sum(y_pred_c * y_true_c, dim=1)  # Shape [B]
    #     # Union = sum(p) + sum(y)
    #     union = torch.sum(y_pred_c + y_true_c, dim=1)  # Shape [B]
    #
    #     # 3. 计算每个样本的 Dice Loss
    #     # Dice_Loss = 1 - (2*I + epsilon) / (U + epsilon)
    #     dice_loss_per_sample = 1.0 - (2.0 * intersection + self.epsilon) / (union + self.epsilon)  # Shape [B]
    #
    #     # 4. 确定有效样本（Ground Truth 中包含该类别的样本）
    #     has_class_gt = torch.sum(y_true_c, dim=1) > self.epsilon  # Shape [B]
    #
    #     # 5. 过滤和求平均
    #     valid_losses = dice_loss_per_sample[has_class_gt]
    #
    #     # 如果 Batch 中没有样本包含该类别 (避免除以 0)
    #     if valid_losses.numel() == 0:
    #         return y_pred_prob.new_tensor(0.0)
    #
    #     # 返回有效样本的平均 Loss
    #     return valid_losses.mean()
