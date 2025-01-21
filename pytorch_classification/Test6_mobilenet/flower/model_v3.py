from functools import partial
from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNActivation(nn.Sequential):
    """
    默认卷积核为3*3, stride=1, padding=1，不改变输入大小
    支持可配置的BN层和激活函数
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 normal_layer=None,
                 activation_layer=None):
        padding = (kernel_size - 1) // 2
        if normal_layer is None:
            normal_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            # groups = 1为普通卷积，groups和输入的channels数相同时为dw卷积
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      groups=groups,
                      bias=False),
            normal_layer(out_channels),
            activation_layer(inplace=True)
        )


class SEBlock(nn.Module):
    def __init__(self, in_channels, squeeze_factor=4):
        squeeze_c = _make_divisible(in_channels // squeeze_factor, 8)
        super(SEBlock, self).__init__()
        # 此处也可以使用切全连接层
        self.fc1 = nn.Conv2d(in_channels, squeeze_c, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_c, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = F.hardsigmoid(self.fc2(scale), inplace=True)
        # 这里得到的scale实际上是通道的权重
        return scale * x


class InvertResidualConfig(object):

    def __init__(self,
                 in_channels: int,
                 kernel_size: int,
                 expand_c: int,
                 out_channels: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 wide_multi: float
                 ):
        """
        :param in_channels:  输入通道数
        :param kernel_size: kernel大小
        :param expand_c: 升维卷积通道数
        :param out_channels: 输出通道数
        :param use_se: 是否使用se
        :param activation: 激活函数
        :param stride: stride
        :param wide_multi:
        """
        self.in_channels = self.adjust_channel(in_channels, wide_multi)
        self.out_channels = self.adjust_channel(out_channels, wide_multi)
        self.kernel_size = kernel_size
        self.expand_c = self.adjust_channel(expand_c, wide_multi)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride

    @staticmethod
    def adjust_channel(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertResidual(nn.Module):
    """
    倒残差结构
    1. 第一层：普通卷积层 1*1, 卷积核个数为tk(为输入channel数的k倍), 输入为(h, w, k), 输出为(h, w, tk)
    2. 第二层: DW卷积 输入为(h, w, tk), 输出为(h/s, t/s, tk), dw卷积只对输入大小进行下采样, 不改变channel数
    3. 第三层: PW卷积 输入为(h/s, t/s, tk), 输出为(h/s, t/s, k')
    """

    def __init__(self,
                 cnf: InvertResidualConfig,
                 normal_layer):
        super(InvertResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.in_channels == cnf.out_channels)
        hidden_channels = cnf.expand_c

        layers = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.in_channels != cnf.expand_c:
            # 1*1升维卷积
            # 注意：当expand_ratio=1时，从输入到depthwise不需要进行升维
            layers.append(ConvBNActivation(cnf.in_channels, hidden_channels,
                                           kernel_size=1,
                                           activation_layer=activation_layer,
                                           normal_layer=normal_layer))

        layers.append(
            # 3*3 DW卷积
            ConvBNActivation(hidden_channels,
                             hidden_channels,
                             kernel_size=cnf.kernel_size,
                             stride=cnf.stride,
                             groups=hidden_channels,
                             activation_layer=activation_layer,
                             normal_layer=normal_layer))

        if cnf.use_se:
            layers.append(SEBlock(hidden_channels))

        layers.append(
            ConvBNActivation(hidden_channels,
                             cnf.out_channels,
                             kernel_size=1,
                             normal_layer=normal_layer,
                             # 注意这里使用identity作为激活函数和无激活函数是一样的
                             activation_layer=nn.Identity)
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, x):
        if self.use_res_connect:
            return self.block(x) + x
        else:
            return self.block(x)


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


class MobileNetV3(torch.nn.Module):
    def __init__(self,
                 inverted_residual_setting,
                 last_channel,
                 num_classes=1000,
                 block=None,
                 norm_layer=None):
        """
        # t, c, n, s
        # t代表扩展因子，即从输入到depthwise会否升维
        # c代表倒残差结构的输出channel
        # n代表bottleneck的重复次数
        # s是步距，只针对第一个倒残差结构
        :param num_classes:
        :param alpha: 代表channel的倍率因子
        """
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # 第一个卷积层，输入为224, 224, 3, 输出为(112, 112, 32)
        first_conv_ouput_channel = inverted_residual_setting[0].in_channels
        layers.append(ConvBNActivation(3,
                                       out_channels=first_conv_ouput_channel,
                                       stride=2,
                                       normal_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layer
        last_conv_input_c = inverted_residual_setting[-1].out_channels
        last_conv_output_c = last_conv_input_c * 6
        layers.append(ConvBNActivation(last_conv_input_c,
                                       last_conv_output_c,
                                       kernel_size=1,
                                       normal_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # last channel 为1280
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_output_c, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(last_channel, num_classes)
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



def mobilenet_v3_small(num_classes=1000, reduced_tail = False):
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """

    with_multi = 1.0
    bneck_conf = partial(InvertResidualConfig, with_multi=with_multi)
    adjust_channel = partial(InvertResidualConfig.adjust_channel, with_multi=with_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    # partial给指定函数添加默认参数值
    bneck_conf = partial(InvertResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertResidualConfig.adjust_channels, width_multi=width_multi)

    # 1. 减少第一个卷积层的卷积核个数
    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


if __name__ == '__main__':
    print(_make_divisible(32 * 1.3, 8))
