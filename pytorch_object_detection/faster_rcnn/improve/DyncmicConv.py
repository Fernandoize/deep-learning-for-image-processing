import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv(nn.Module):
    """
    动态卷积基的数量：初始设置为4-8个基，根据任务复杂度调整。过多的基会增加计算量，过少则可能限制表达能力。
    使用一个小型网络（如MLP）生成动态卷积核权重 将生成的卷积核应用于输入特征图，得到分类和回归结果

    动态卷积生成器：生成动态卷积核权重。
    动态卷积操作：将生成的卷积核应用于输入特征图。
    分类与回归头：输出目标类别和边界框偏移量。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, num_bases=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_bases = num_bases  # 动态卷积基的数量

        # 动态卷积基生成器
        self.base_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_bases * in_channels * kernel_size**2, 1),
            nn.Unflatten(1, (num_bases, in_channels, kernel_size, kernel_size))
        )

        # 注意力权重生成器
        self.attn_generator = nn.Sequential(
            nn.Conv2d(in_channels, num_bases, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 生成动态卷积基
        bases = self.base_generator(x)  # (B, num_bases, in_channels, k, k)
        B, num_bases, in_channels, k, _ = bases.shape

        # 生成注意力权重
        attn = self.attn_generator(x)  # (B, num_bases, H, W)
        attn = attn.view(B, num_bases, -1)  # (B, num_bases, H*W)

        # 动态卷积操作
        x_unfold = F.unfold(x, self.kernel_size, padding=self.kernel_size//2)  # (B, in_channels*k*k, H*W)
        x_unfold = x_unfold.view(B, in_channels, self.kernel_size**2, -1)  # (B, in_channels, k*k, H*W)
        x_unfold = x_unfold.permute(0, 3, 1, 2)  # (B, H*W, in_channels, k*k)

        # 加权求和动态卷积基
        dynamic_kernels = torch.einsum('bnihw,bn->bihw', bases, attn.mean(dim=-1))  # (B, in_channels, k, k)
        dynamic_kernels = dynamic_kernels.view(B, in_channels, -1)  # (B, in_channels, k*k)

        # 应用动态卷积
        output = torch.einsum('bhik,bik->bh', x_unfold, dynamic_kernels)  # (B, H*W, out_channels)
        output = output.view(B, -1, x.shape[2], x.shape[3])  # (B, out_channels, H, W)

        return output


class FastRCNNWithDynamicConv(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))  # RoI Pooling
        self.dynamic_conv = DynamicConv(in_channels=256, out_channels=1024)  # 动态卷积
        self.cls_head = nn.Linear(1024, num_classes)  # 分类头
        self.reg_head = nn.Linear(1024, 4 * num_classes)  # 回归头

    def forward(self, x, rois):
        # 提取特征
        features = self.backbone(x)
        # RoI Pooling
        pooled_features = self.roi_pool(features[rois])
        # 动态卷积
        dynamic_features = self.dynamic_conv(pooled_features)
        # 分类与回归
        cls_scores = self.cls_head(dynamic_features.mean(dim=(2, 3)))
        reg_preds = self.reg_head(dynamic_features.mean(dim=(2, 3)))
        return cls_scores, reg_preds