# create model
# 此处不需要构建辅助分类器
import os

import torch
from matplotlib import pyplot as plt
from torchinfo import summary
import matplotlib.pyplot as plt
import torchvision.models as models
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

from pytorch_classification.Test4_googlenet.flower.model import GoogleNet, BasicConv2d


# def visual_layers():
#     conv1 = dict(model.named_children())['conv1']
#     if isinstance(conv1, BasicConv2d):
#         conv1 = dict(conv1.named_children())['conv']
#     kernel_set = conv1.weight.detach()
#     num = len(conv1.weight.detach())
#     print(kernel_set.shape)
#     for i in range(0, num):
#         if i > 3:
#             break
#         i_kernel = kernel_set[i]
#         plt.figure(figsize=(20, 17))
#         if (len(i_kernel)) > 1:
#             for idx, filer in enumerate(i_kernel):
#                 plt.subplot(9, 9, idx + 1)
#                 plt.axis('off')
#                 plt.imshow(filer[:, :].detach(), cmap='bwr')
#         plt.show()


import matplotlib.pyplot as plt
import torchvision.models as models
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

model = models.alexnet(pretrained=True)
backprop = Backprop(model)

image = load_image('./bustard.jpeg')
owl = apply_transforms(image)

target_class = 24
backprop.visualize(owl, target_class, guided=True, use_gpu=True)