import datetime
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class CLIPLikeModel(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        # 加载预训练的ResNet18
        resnet = models.resnet18(pretrained=True)

        # 保留ResNet的各层但移除最终的全连接层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 投影层 - 将最后两层特征投影到相同的维度空间
        # layer3输出通道为256，layer4输出通道为512
        self.image_projection = nn.Linear(512, projection_dim)
        self.text_projection = nn.Linear(256, projection_dim)

        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        # 依次通过ResNet的各个层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # 获取layer3的输出（作为"text" encoder的特征）
        layer3_output = self.layer3(x)

        # 获取layer4的输出（作为"image" encoder的特征）
        layer4_output = self.layer4(layer3_output)

        # 全局平均池化
        layer3_features = F.adaptive_avg_pool2d(layer3_output, 1).squeeze(-1).squeeze(-1)
        layer4_features = F.adaptive_avg_pool2d(layer4_output, 1).squeeze(-1).squeeze(-1)

        # 投影到共同的特征空间
        image_features = self.image_projection(layer4_features)
        text_features = self.text_projection(layer3_features)

        # 归一化特征
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        return image_features, text_features


# 对比学习损失函数
def clip_loss(image_features, text_features, logit_scale):
    # 计算相似度矩阵
    logits = logit_scale * image_features @ text_features.T

    # 创建标签 (对角线为正样本)
    labels = torch.arange(logits.shape[0], device=logits.device)

    # 计算交叉熵损失 (image-to-text和text-to-image)
    image_loss = F.cross_entropy(logits, labels)
    text_loss = F.cross_entropy(logits.T, labels)

    # 总损失是两个方向损失的平均
    loss = (image_loss + text_loss) / 2
    return loss



def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model num_classes equal background + 20 classes
    model = CLIPLikeModel()
    # print(model)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    # 训练循环示例

    best_validation = 0
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss = train_one_epoch(model, train_loader, optimizer, device=device, epoch=epoch)
        # 验证
        val_metrics = validate(model, validate_loader, device)

        # 记录当前性能
        current_performance = val_metrics['mean_retrieval']['R@1']

        print(f"Epoch {epoch + 1}/{args.epochs}:")
        print(f"  Train Loss: {mean_loss:.4f}")
        print(f"  Validation R@1: {val_metrics['image_to_text']['R@1']:.2f}% (Image→Text), "
              f"{val_metrics['text_to_image']['R@1']:.2f}% (Text→Image), "
              f"{val_metrics['mean_retrieval']['R@1']:.2f}% (Mean)")

        # 保存最佳模型
        if current_performance > best_validation:
            best_validation = current_performance
            torch.save(model.state_dict(), 'best_clip_like_model.pt')
            print(f"  New best model saved! Mean R@1: {current_performance:.2f}%")
        # learning_rate.append(lr_scheduler)
        # # update the learning rate
        lr_scheduler.step()
        # # save weights
        # save_files = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'lr_scheduler': lr_scheduler.state_dict(),
        #     'epoch': epoch}
        # if args.amp:
        #     save_files["scaler"] = scaler.state_dict()
        # torch.save(save_files, "./save_weights/clip-resnet-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    train_bar = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        # 将同一批次的图像通过模型，但把layer3和layer4的特征分别视为"image"和"text"特征
        images, labels = data
        image_features, text_features = model(images.to(device))

        # 计算损失
        loss = clip_loss(image_features, text_features, model.logit_scale)
        loss.backward()
        accu_loss += loss.detach()
        train_bar.desc = "[train epoch {}] loss: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # 反向传播
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1)


def validate(model, val_dataloader, device):
    model.eval()
    all_image_features = []
    all_text_features = []

    val_bar = tqdm(val_dataloader, file=sys.stdout)

    with torch.no_grad():
        for step, data in enumerate(val_bar):
            images, labels = data
            images = images.to(device)

            # 前向传播获取两层特征
            image_features, text_features = model(images)

            all_image_features.append(image_features)
            all_text_features.append(text_features)

    # 连接所有批次的特征
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    # 计算相似度矩阵
    similarity = all_image_features @ all_text_features.T

    # 评估检索性能
    image_to_text_retrieval = evaluate_retrieval(similarity)
    text_to_image_retrieval = evaluate_retrieval(similarity.T)

    # 正确计算平均指标
    mean_retrieval = {
        'R@1': (image_to_text_retrieval['R@1'] + text_to_image_retrieval['R@1']) / 2,
        'R@5': (image_to_text_retrieval['R@5'] + text_to_image_retrieval['R@5']) / 2,
        'R@10': (image_to_text_retrieval['R@10'] + text_to_image_retrieval['R@10']) / 2,
        'mean_rank': (image_to_text_retrieval['mean_rank'] + text_to_image_retrieval['mean_rank']) / 2
    }

    return {
        'image_to_text': image_to_text_retrieval,
        'text_to_image': text_to_image_retrieval,
        'mean_retrieval': mean_retrieval
    }

def evaluate_retrieval(similarity_matrix):
    """评估检索性能，计算R@1, R@5, R@10指标"""
    ranks = []
    num_samples = similarity_matrix.shape[0]

    # 对于每个查询
    for i in range(num_samples):
        # 获取相似度得分
        similarities = similarity_matrix[i]

        # 目标是对角线元素(即正确匹配)
        target_similarity = similarities[i]

        # 计算排名 (有多少个元素的相似度比目标高)
        rank = (similarities > target_similarity).sum().item() + 1
        ranks.append(rank)

    # 计算召回率
    r1 = sum(rank == 1 for rank in ranks) / len(ranks)
    r5 = sum(rank <= 5 for rank in ranks) / len(ranks)
    r10 = sum(rank <= 10 for rank in ranks) / len(ranks)

    return {
        'R@1': r1 * 100,  # 转换为百分比
        'R@5': r5 * 100,
        'R@10': r10 * 100,
        'mean_rank': sum(ranks) / len(ranks)
    }


if __name__ == "__main__":
    """
    nohup python3 change_backbone_with_fpn.py --batch_size 8 > output.log 2>&1 &
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--model_name', default='resnet', help='model_name')
    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='../..', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=4, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)