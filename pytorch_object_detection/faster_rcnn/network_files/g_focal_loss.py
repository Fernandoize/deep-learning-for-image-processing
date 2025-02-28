import torch
import torch.nn as nn
import torch.nn.functional as F


class GFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=1.0):
        super().__init__()
        self.alpha = alpha  # QFL 的正负样本权重
        self.gamma = gamma  # Focal Loss 的调制因子
        self.beta = beta  # QFL 的质量调制因子

    def quality_focal_loss(self, pred, target, iou_targets):
        # pred: [N, num_classes], 预测概率 (sigmoid 后)
        # target: [N], 类别标签
        # iou_targets: [N], 真实 IoU 分数 (0~1)

        # 前景样本 mask
        target = torch.cat(target, dim=0)
        iou_targets = torch.cat(iou_targets, dim=0)
        mask = target > 0
        pred = pred.sigmoid()  # 转换为概率

        # 计算 QFL
        qfl_loss = torch.zeros_like(pred)
        for i in range(pred.size(1)):  # 遍历每个类别
            p = pred[:, i]
            y = (target == i).float() * iou_targets  # 目标质量分数
            qfl_loss[:, i] = -torch.abs(y - p).pow(self.beta) * (
                    y * torch.log(p + 1e-12) + (1 - y) * torch.log(1 - p + 1e-12)
            )
        return qfl_loss.sum() / mask.sum().clamp(min=1)

    def distribution_focal_loss(self, pred, target):
        # pred: [N, 4, num_bins], 预测分布 logits
        # target: [N, 4], 真实边界框偏移量

        target = torch.cat(target, dim=0)
        num_bins = pred.size(-1)
        bin_range = torch.arange(num_bins, dtype=torch.float32, device=pred.device)

        loss = 0
        for i in range(4):  # 对 dx, dy, dw, dh 分别计算
            p = F.softmax(pred[:, i], dim=-1)  # [N, num_bins]
            y = target[:, i]  # [N]
            y_normalized = y * (num_bins - 1)  # 归一化到 [0, num_bins-1]

            left = y_normalized.floor().long()
            right = left + 1
            left_weight = right.float() - y_normalized
            right_weight = y_normalized - left.float()

            left = left.clamp(0, num_bins - 1)
            right = right.clamp(0, num_bins - 1)

            loss += -(
                    left_weight * torch.log(p[range(len(y)), left] + 1e-12) +
                    right_weight * torch.log(p[range(len(y)), right] + 1e-12)
            )
        return loss.sum() / len(target)

    def forward(self, cls_pred, reg_pred, cls_target, reg_target, iou_targets):
        qfl_loss = self.quality_focal_loss(cls_pred, cls_target, iou_targets)
        dfl_loss = self.distribution_focal_loss(reg_pred, reg_target)
        return qfl_loss, dfl_loss