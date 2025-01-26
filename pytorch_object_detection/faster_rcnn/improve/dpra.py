import torch
from torchvision.ops import roi_align
from torch import nn

from torch.nn import functional as F


class DRPA(nn.Module):
    """
    在Fast R-CNN的RoI Pooling前加入可学习的注意力门控，根据区域提议（Region Proposal）的上下文动态调整特征权重。
    通过计算提议区域与全局特征的相似度，强化目标区域的响应。
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.query = nn.Conv2d(feat_dim, feat_dim, 1)
        self.key = nn.Conv2d(feat_dim, feat_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, features, proposals):
        # 提取提议区域特征
        rois = roi_align(features, proposals, output_size=7)
        # 计算注意力权重
        query = self.query(rois).mean(dim=(2,3))
        key = self.key(features).mean(dim=(2,3))
        attn = torch.matmul(query, key.transpose(1,2))
        attn = F.softmax(attn, dim=-1)
        # 特征增强
        return features + self.gamma * torch.matmul(attn, features)