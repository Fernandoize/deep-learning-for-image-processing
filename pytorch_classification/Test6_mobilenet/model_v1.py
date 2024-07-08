from torch import nn


import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class MobileNetV1(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False, pretrained=False):
        super().__init__()
        self.features = features
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        if self.init_weights:
            self._initialize_weights()

        if pretrained:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 1000),
            )
            self.load_state_dict(torch.load("vgg16-397923af.pth"), strict=True)
            for name, param in self.named_parameters():
                param.requires_grad = False
            self.classifier = nn.Sequential(*list(self.classifier.children()),
                                            nn.ReLU(True),
                                            nn.Dropout(0.5),
                                            nn.Linear(1000, num_classes))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# 卷积层越大，感受野越大
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, groups=v)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


def mobile_net_v1(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = MobileNetV1(make_layers(cfg), **kwargs)
    return model

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


classes = {}
with open('imagenet_1k_class.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(":")
        classes[line[0]] = line[1]


if __name__ == '__main__':
    mobile_net_v1 = mobile_net_v1(model_name="vgg16", num_classes=1000, init_weights=True)
    mobile_net_v1.load_state_dict(torch.load("./vgg16-397923af.pth"))
    mobile_net_v1.eval()
    img = Image.open("./bustard.jpeg")
    img = transform(img).unsqueeze(0)
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        predict_y = mobile_net_v1(img)
        predict_y = softmax(predict_y)
        predict_y = torch.max(predict_y, 1)[1]
        print(f'class: {classes.get(str(predict_y.item()))}, {predict_y.item()}')

