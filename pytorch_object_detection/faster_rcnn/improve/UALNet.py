from torch import nn


class UCCM(nn.Module):
    """
    基于MobileNetV3，在倒残差结构中嵌入水下色彩校正模块（UCCM），自动补偿蓝绿色偏。
    使用深度可分离卷积减少计算量，适应边缘设备部署。
    兼顾计算效率与水下色彩恢复能力。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.color_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 3, 1),  # 预测RGB三通道增益
            nn.Sigmoid()
        )

    def forward(self, x):
        color_weights = self.color_net(x)
        return x * color_weights  # 色彩校正