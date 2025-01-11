import torch
from numpy.matlib import identity
from torch import nn
from torchinfo import summary


# ***************************#
# 网络结构层数都是4层

# **************************#

class BasicBlock(nn.Module):
    """
    针对18和34层的残差结构
    expansion = 1 代表残差结构中第一层和第二层卷积核的个数是否发生变化
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        1. 实线：卷积核个数不变，特征shape不变
        2. 虚线：卷积核个数*2， 特征shape / 2, stride = 2
        2. downsample代表虚线的残差结构，conv 1*1, stride = 2
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 使用bn时，bias设置和不设置效果一样
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

        # relu无参数，可以共用
        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            # downsample
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 注意: 先residual, 再relu
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    针对50和101、152层的残差结构
    expansion = 4 代表残差结构中第二层和第三层卷积核的个数发生4倍变化
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        """
        1. 实线：卷积核个数第二到三层增加4倍，第一到第二层不变，特征shape不变
        2. 虚线：卷积核个数*2， 特征shape / 2, stride = 2
        """

        # 1*1降维
        self.conv1= nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #--------------
        self.cov2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # --------------
        # 第三层的卷积核个数扩张四倍
        self.cov3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.cov3(out)
        out = self.bn3(out)
        # 注意: 先residual, 再relu
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        """

        Args:
            block: block对应残差结构，18/34层对应BasicBlock, 50和101、152层对应BottleNeck
            blocks_num: 每层残差结构的数量[2, 2, 2, 2]
            num_classes:
            init_weights:
        """
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.include_top = include_top

        # 输入为(3, 224, 224)， 输出为(64, 112, 112)
        self.input_channel = 64
        # 踩坑，卷积层的key需要和原文一样，否则无法加载预训练权重
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input_channel, kernel_size=7, stride=2,
                      padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_channel)
        self.relu = nn.ReLU(inplace=True)

        # 输入为(64, 112, 112), 输出为(64, 56, 56)
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 18/34层输入为(64, 56, 56), 输出为(64, 28, 28)
        # 50/101/152层输入为(64, 56, 56), 输出为(256, 28, 28)
        # conv2_x不需要下采样
        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=2)

        # 18/34层输入为(64, 28, 28), 输出为(128, 14, 14)
        # 50/101/152层输入为(256, 28, 28), 输出为(512, 14, 14)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)

        # 输入为(block_channel / 2, 28, 28), 输出为(block_channel * expansion, 14, 14)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)

        # 输入为(block_channel / 2, 14, 14), 输出为(block_channel * expansion, 7, 7)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # 这里是为了只使用resnet作为卷积backbone
        if self.include_top:
            # ? 这里采用1*1会丢失信息吗？
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x =  self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, block_channel, block_num, stride=1):
        """
        Args:
            block:
            block_channel:
            block_num:
            stride: stride 参数在conv2_x = 1, 在conv3_x-conv5_x = 2, 需要下采样

        Returns:

        """
        downsample = None

        # 对于18/34层conv_2_x, input_channel = 64, block_channel = 64

        # 对于18/34层conv3_x, input_channel = 64, block_channel = 128
        # 对于50/101/152层 conv_2_x, input_channel = 64, block_channel = 64, expansion = 4
        # 对于50/101/152层 conv_3_x, input_channel = 256, block_channel = 128, expansion = 4
        # stride != 1 代表特征图发生变化, self.input_channel != block_channel * block.expansion 代表深度发生变化
        out_channel = block_channel * block.expansion
        if stride != 1 or self.input_channel != out_channel:
            # 注意 block_channel*block.expansion，残差下采样卷积核的个数与最后一层卷积核心的个数一致，所以需要乘expansion
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.input_channel, out_channels=out_channel,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )

        layers = [block(
            in_channels=self.input_channel,
            out_channels=block_channel,
            stride=stride,
            downsample=downsample,
        )]

        # 第一个残差块: 对于18/34层conv_2_x，channel没有变化和特征无变化，对于50/101/152层，channel发生变化
        # 对于conv3_x-conv5_x，channel和特征都发生了变化

        # 对于18/34 conv2_2 channel无变化, 对于50/101/152层， conv2_3，channel扩大四倍
        self.input_channel = out_channel

        # convx_x中从第2个残差块起, 都是实线连接，没有channel和特征大小的变化，不需要下采样
        for i in range(1, block_num):
            layers.append(block(
                in_channels=self.input_channel,
                out_channels=block_channel
            ))

        return nn.Sequential(*layers)


def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnet152(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


if __name__ == '__main__':
    resnet = resnet18(num_classes=5, include_top=True)
    input = torch.randn(1, 3, 224, 224)
    print(summary(resnet, (1, 3, 224, 224)))
    # print(resnet(input))