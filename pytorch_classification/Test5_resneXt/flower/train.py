import json
import os
import random
import sys

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from torchvision import transforms, datasets
from tqdm import tqdm

from pytorch_classification.Test5_resnet.flower.model import resnet34

# 初始化wandb
wandb.init(project='resnet34')

# 超参数设置
config = wandb.config
config.batch_size = 32
config.test_batch_size = 32
config.epochs = 50
config.lr = 0.001
config.momentum = 0.9
config.weight_decay = 0.0002
config.use_cuda = True
config.seed = 2043
config.log_interval = 10
config.architecture = "resnet34"

# 设置随机数
def set_seed():
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    numpy.random.seed(config.seed)


def main():
    # 1. device设置
    is_cuda_available = torch.cuda.is_available()
    is_mps_available = torch.backends.mps.is_available()
    use_cuda = config.use_cuda and (is_cuda_available or is_mps_available)
    device = torch.device("mps" if is_mps_available else "cpu")
    device = torch.device("cuda:0") if is_cuda_available else device
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    print("using {} device.".format(device))

    set_seed()

    # 2. 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))  # get data root path
    image_path = os.path.join(data_root, "data_set")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 3. 数据集加载
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size, shuffle=True,
                                               **kwargs)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=config.test_batch_size, shuffle=False,
                                                  **kwargs)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 4. 模型定义
    # 注意：为了正确加载权重，这里不要设置classes
    net = resnet34(num_classes=5)
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load("resnet34.pth", map_location=device),
    #                                                     strict=False)
    # print("missing keys: {}".format(missing_keys))
    # print("unexpected keys: {}".format(unexpected_keys))

    # 1. 迁移学习的第一种方法
    # 全连接层的输入特征shape
    # inchannel = net.fc.in_features
    # net.fc = nn.Linear(inchannel, 5)
    net.to(device)

    # 4. 损失函数+ 优化器
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # 5. 训练
    epochs = config.epochs
    save_path = './resnet34.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    # # wandb.log用来记录一些日志(accuracy,loss and epoch), 便于随时查看网路的性能
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            # 注意：这里有两个辅助分类器的输出
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        val_loss = 0.0
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))

                predict_y = torch.max(outputs, dim=1)[1]

                val_loss += loss.item()
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[valid epoch %d] train_loss: %.3f, test_loss: %.3f, val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_loss / val_steps, val_accurate))

        # 使用wandb.log 记录你想记录的指标
        wandb.log({
            "Train Loss": running_loss / train_steps,
            "Val Accuracy": 100. * val_accurate,
            "Val Loss": val_loss / val_steps
        })

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('Finished Training')


if __name__ == '__main__':
    main()
