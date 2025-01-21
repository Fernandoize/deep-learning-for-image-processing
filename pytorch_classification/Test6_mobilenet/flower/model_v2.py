from torch import nn
import torch

class ConvBNReLU(nn.Sequential):
    """
    默认卷积核为3*3, stride=1, padding=1，不改变输入大小
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            # groups = 1为普通卷积，groups和输入的channels数相同时为dw卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertResidual(nn.Module):
    """
    倒残差结构
    1. 第一层：普通卷积层 1*1, 卷积核个数为tk(为输入channel数的k倍), 输入为(h, w, k), 输出为(h, w, tk)
    2. 第二层: DW卷积 输入为(h, w, tk), 输出为(h/s, t/s, tk), dw卷积只对输入大小进行下采样, 不改变channel数
    3. 第三层: PW卷积 输入为(h/s, t/s, tk), 输出为(h/s, t/s, k')
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertResidual, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        hidden_channels = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            # 1*1升维卷积
            # 注意：当expand_ratio=1时，从输入到depthwise不需要进行升维
            layers.append(ConvBNReLU(in_channels, hidden_channels, kernel_size=1))

        layers.extend([
            # 3*3 DW卷积
            ConvBNReLU(hidden_channels, hidden_channels, stride=stride, groups=hidden_channels),
            # 1*1 PointWise卷积
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # 注意：此处为线性激活函数，也就是不添加激活函数
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        else:
            return self.conv(x)


def _make_divisible(ch, divisor=8, min_ch=None):
    """
     向下取整寻找离8最近的数字, 可能有利于GPU底层运算
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        """

        # t, c, n, s
        # t代表扩展因子，即从输入到depthwise会否升维
        # c代表倒残差结构的输出channel
        # n代表bottleneck的重复次数
        # s是步距，只针对第一个倒残差结构
        :param num_classes:
        :param alpha: 代表channel的倍率因子
        """
        block = InvertResidual
        input_channel = _make_divisible(32 * alpha, 8)
        last_channel = _make_divisible(1280 * alpha, 8)

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]


        features = []
        # 第一个卷积层，输入为224, 224, 3, 输出为(112, 112, 32)
        features.append(ConvBNReLU(3, out_channels=input_channel, stride=2))

        # 中间的倒残差结构, 输入为(112, 112, 32), 输出为(7, 7, 320)
        for t, c, n, s in inverted_residual_setting:
            output_channels = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                # 第一个倒残差结构不需要升维度
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channels, stride, expand_ratio=t))
                input_channel = output_channels


        # 最后一个卷积层, 输入为(7, 7, 320), 输出为(7, 7, 1280)
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        self.features = torch.nn.Sequential(*features)

        # 卷积层以便于适应不同大小的图片, 输入为(7, 7, 1280), 输出为(1, 1, 1280)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        # classifier, 输入为(1, 1, 1280), 输出为(num_classes)
        self.classifier = torch.nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
            # 也可以换为1*1卷积
           # nn.Conv2d(last_channel, num_classes, 1)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    print(_make_divisible(32 * 1.3, 8))