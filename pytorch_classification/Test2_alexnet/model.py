import torch.nn as nn
from torchinfo import summary
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from feature_pyramid_network import BackboneWithFPN


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


class AlexNetFPN(nn.Module):
    def __init__(self, num_classes=5, init_weights=True, **kwargs):
        super(AlexNetFPN, self).__init__()
        self.alex_net = AlexNet(num_classes=num_classes, init_weights=init_weights)

        return_layers = {
            "features.0": "0",
            "features.3": "1",
            "features.6": "2",
            "features.8": "3",
            "features.10": "4"
        }  # stride 32
        # 提供给fpn的每个特征层channel
        name_modules = dict(self.alex_net.named_modules())
        in_channels_list = [name_modules[layer_name].out_channels for layer_name in return_layers.keys()]
        new_backbone = create_feature_extractor(self.alex_net, return_layers)
        self.backbone_with_fpn = BackboneWithFPN(new_backbone,
                                            return_layers=return_layers,
                                            in_channels_list=in_channels_list,
                                            out_channels=128,
                                            extra_blocks=LastLevelMaxPool(),
                                            re_getter=False)

    def forward(self, x):
        x = self.backbone_with_fpn(x)
        x = x['0']
        x = self.alex_net.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.alex_net.classifier(x)
        return x


if __name__ == '__main__':
    """
    Params 14,591,685
    Linear占比: 90%
    58.37M
    """
    model = AlexNetFPN(num_classes=5, init_weights=True)
    input = torch.randn(1, 3, 224, 224)
    summary(model, (1, 3, 224, 224))

    # 第一种融合：从高到低融合，取低层或中层
    # 第二种融合：从低到高融合，取高层或中层
    # 第三种融合：双向融合，取高、中、低层

    # 第四种融合方式：加法融合，拼接融合 理论来源
    #