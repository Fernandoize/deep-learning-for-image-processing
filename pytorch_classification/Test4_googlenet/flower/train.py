import os
import random
import sys
import json

import numpy
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from pytorch_classification.Test4_googlenet.flower.model import GoogleNet

# 初始化wandb
import wandb
wandb.init(project='googleLet')

# 超参数设置
config = wandb.config  # config的初始化
config.batch_size = 32
config.test_batch_size = 32
config.epochs = 10
config.lr = 0.0003
config.momentum = 0.1
config.use_cuda = False
config.seed = 2043
config.log_interval = 10

# 设置随机数
def set_seed():
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    numpy.random.seed(config.seed)

def main():
    use_cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print("using {} device.".format(device))

    set_seed()

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
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

    nw = min([os.cpu_count(), config.batch_size if config.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

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

    net = GoogleNet(num_classes=5, aux_logits=True, init_weights=True)
    net.to(device)
    net.load_state_dict(torch.load('./googleNet.pth'))
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr)

    epochs = 10
    save_path = './googleNet.pth'
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
            logits, aux2_logits, aux1_logits = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux1_logits, labels.to(device))
            loss2 = loss_function(aux2_logits, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3

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
            wandb.save(save_path)
    print('Finished Training')


if __name__ == '__main__':
    main()
