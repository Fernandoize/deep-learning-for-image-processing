import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


# 定义 UCCM 模块
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
        self.channel_expand = nn.Conv2d(3, in_channels, 1)  # 通道扩展

    def forward(self, x):
        color_weights = self.color_net(x)
        color_weights = self.channel_expand(color_weights)  # (B, in_channels, 1, 1)
        return x * color_weights  # 色彩校正


if __name__ == '__main__':

    # 加载预训练的 MobileNetV3 Small
    model = mobilenet_v3_small(pretrained=True)

    # 动态插入 UCCM 模块
    uccm_layers = nn.ModuleList([UCCM(in_channels=model.features[4].out_channels)])  # 根据目标层的输出通道数初始化 UCCM

    # 替换目标层
    model.features[4] = nn.Sequential(model.features[4], uccm_layers[0])

    # 测试模型
    x = torch.randn(1, 3, 224, 224)  # 输入图像
    output = model(x)
    print(output.shape)  # 输出形状