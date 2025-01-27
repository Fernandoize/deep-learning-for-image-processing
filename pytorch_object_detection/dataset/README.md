## 类别

1. 海参
2. 扇贝
3. 海胆
4. 海星


## 数据统计

train size: 5448, 347个无object
valid size: 1556, 86个无object
test size: 778, 53个无object

图片大小640 * 640 * 3

train_size: 5448, class_stats: defaultdict(<class 'int'>, {'holothurian': 5361, 'echinus': 31887, 'starfish': 9354, 'scallop': 1260})
val_size: 1556, class_stats: defaultdict(<class 'int'>, {'holothurian': 1473, 'echinus': 9294, 'starfish': 2762, 'scallop': 393})
test_size: 778, class_stats: defaultdict(<class 'int'>, {'holothurian': 713, 'echinus': 4436, 'starfish': 1296, 'scallop': 182})


## 改进点
1. 先尝试使用MobileNetV2 + FPN
1. 注意力机制 通道注意力机制SENet, 空间注意力机制：


## 结论：使用mobilenetv3 large和resnet34效果差不多，使用fpn效果会更好，但相比pascal voc 数据效果比较差，主要是数据的问题

## 使用mobilenet v3进行测试





