import torch.nn as nn
import torch
from torchinfo import summary


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetFPN(AlexNet):
    def __init__(self, num_classes=1000):
        super(AlexNetFPN, self).__init__()

        # Lateral connections for FPN
        # 1 * 1 卷积降维操作
        self.lateral_5 = nn.Conv2d(128, 128, kernel_size=1)  # For last layer
        self.lateral_4 = nn.Conv2d(192, 128, kernel_size=1)  # For 4th layer
        self.lateral_3 = nn.Conv2d(192, 128, kernel_size=1)  # For 3rd layer
        self.lateral_2 = nn.Conv2d(128, 128, kernel_size=1)  # For 3rd layer
        self.lateral_1 = nn.Conv2d(48, 128, kernel_size=1)  # For 3rd layer

        # FPN output layers
        self.fpn_5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fpn_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fpn_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fpn_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fpn_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
            高层特征抽象含义更多，但分辨率低，通过特征金字塔，将高分辨率和高层特征结合起来
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # Store intermediate features
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [0, 3, 6, 8, 10]:  # Store features after 3rd, 4th, and 5th conv layers
                features.append(x)

        # Build FPN features
        c1, c2, c3, c4, c5 = features

        # FPN top-down pathway and lateral connections
        p5 = self.lateral_5(c5)
        p4 = self._upsample_add(p5, self.lateral_4(c4))
        p3 = self._upsample_add(p4, self.lateral_3(c3))
        p2 = self._upsample_add(p3, self.lateral_2(c2))
        p1 = self._upsample_add(p2, self.lateral_1(c1))

        # FPN output layers
        p5 = self.fpn_5(p5)
        p4 = self.fpn_4(p4)
        p3 = self.fpn_3(p3)
        p2 = self.fpn_2(p2)
        p1 = self.fpn_1(p1)


        # Use p5 for classification (you could modify this to use all pyramid levels)
        x = self.avgpool(p1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # Return all pyramid features and classification output
        return {
            'out': x,
            'features': [p1, p2, p3, p4, p5]
        }



if __name__ == '__main__':
    """
    Params 14,591,685
    Linear占比: 90%
    58.37M
    """
    model = AlexNet(num_classes=5, init_weights=True)
    input = torch.randn(1, 3, 224, 224)
    summary(model, (1, 3, 224, 224))