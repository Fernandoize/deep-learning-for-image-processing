import torch
from PIL.Image import Image
from torch import nn
import torch.nn.functional as F
from torchinfo import summary


class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet, self).__init__()
        self.aux_logits = aux_logits

        # 输入为(224, 224, 3), 输出为(112, 112, 64) 这里padding 应该是左2右3，为了方便直接两边都3，然后pytorch默认向下取整112
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # 输出为(112, 112, 64), 输出为(56, 56, 64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 注意： 这里省略了LocalResponseNormal
        # 输入为(56, 56, 64), 输出为(56, 56, 64)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        # 输入为(56, 56, 64), 输出为(56, 56, 192)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        # 输入为(56, 56, 64), 输出为(28, 28, 192)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 输入为(28, 28, 192), 输出为(28, 28 256) 256  = 64 + 128 + 32 + 32
        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # 输入为(28, 28, 256), 输出为(28, 28, 480)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # 输入为(28, 28, 480), 输出为(14, 14, 480)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 输入为(14, 14, 480), 输出为(14, 14, 512)
        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # 输入为(14, 14, 512), 输出为(14, 14, 512)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # 输入为(14, 14, 512), 输出为(14, 14, 512)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # 输入为(14, 14, 512), 输出为(14, 14, 528)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # 输入为(14, 14, 528), 输出为(14, 14, 832)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        # 输入为(14, 14, 832), 输出为(7, 7, 832)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 输入为(7, 7, 832), 输出为(7, 7, 832)
        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        # 输入为(7, 7, 832), 输出为(7, 7, 1024)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)
        # 输入为(7, 7, 1024), 输出为(1, 1, 1024)
        # AvgPool2d 要求输入大小必须为224 * 224, 才能得到1 * 1的输出
        # self.avg_pool_5 = nn.AvgPool2d(7, stride=1)
        # 自适应的池化层可以输入任何大小的特征图
        self.avg_pool_5 = nn.AdaptiveAvgPool2d((1, 1))

        # inception_4a 和 inception_4d后面加增强分类分支
        if self.training and aux_logits:
            self.aux1 = InceptionAux(512)
            self.aux2 = InceptionAux(528)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool2(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool3(x)

        x = self.inception_4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception_4e(x)
        x = self.max_pool4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pool_5(x)

        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        # N x 1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class InceptionAux(nn.Module):
    """辅助分类器"""
    def __init__(self, in_channels, num_classes=1000):
        super(InceptionAux, self).__init__()
        # 这里也是输出到全连接层时 输出特征大小为4 * 4
        # 输入为(14, 14), 输出为(4, 4)
        # self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(4 * 4 * 128, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14, 输出为(512, 4, 4), (528, 4, 4)
        x = self.avg_pool(x)
        # 输入为输出为(512, 4, 4), (528, 4, 4), 输出为(128, 4, 4)
        x = self.conv(x)

        # 输出为N * 2048
        x = torch.flatten(x, start_dim=1)
        # 采用70%没有50%效果好, self.train = True 当model.train()时，self.train = False当model.eval()时
        x = F.dropout(x, 0.5, training=self.training)

        # 输出为N * 1024
        x = F.relu(self.fc1(x), inplace=False)
        x = F.dropout(x, 0.5, training=self.training)

        # 输出为N * num_classes
        return self.fc2(x)


class Inception(nn.Module):
    """
    Inception
    1 * 1卷积核
    3 * 3卷积核 padding = 1
    5 * 5卷积核 padding = 2
    Pooling 3 * 3， padding = 1
    以上卷积核均不改变特征图的宽和高
    """
    def __init__(self,  in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj, **kwargs):
        """
        :param in_channels:
        :param ch1x1:  1*1卷积核个数
        :param ch3x3_reduce: 3*3卷积之前的1*1降维卷积核个数
        :param ch3x3: 3*3卷积核个数
        :param ch5x5_reduce: 3*3卷积之前的1*1降维卷积核个数
        :param ch5x5: 5*5卷积核个数
        :param pool_proj: 3*3最大池化之后的1*1降维卷积核个数
        :param kwargs:
        """
        super(Inception, self).__init__()
        # inception 第一个分支
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # inception 第二个分支
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_reduce, kernel_size=1),
            BasicConv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
        )

        # inception 第三个分支
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5_reduce, kernel_size=1),
            BasicConv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
        )

        # inception 第四个分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )


    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


if __name__ == '__main__':
    """
    Params 12,358,405
    Linear占比: 90%
    模型大小：23.91M
    """
    google_net = GoogleNet(num_classes=5, aux_logits=True, init_weights=True)
    google_net.train()
    input = torch.randn(1, 3, 224, 224)
    summary(google_net, (1, 3, 224, 224))