from torch import nn


class MSCA(nn.Module):
    """
    在主干网络输出的不同层级特征图（如ResNet的conv3_x、conv4_x）中，
    分别嵌入**空间注意力（SA）和通道注意力（CA）**模块。
    通过加权融合多尺度注意力权重，抑制背景噪声，增强生物轮廓特征。
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 通道注意力（Squeeze-and-Excitation）
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力（Spatial Attention）
        self.sa = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca_weight = self.ca(x)
        sa_weight = self.sa(x)
        # 协同融合
        return x * ca_weight * sa_weight

