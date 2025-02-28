
import json
from collections import defaultdict

from matplotlib import pyplot as plt

if __name__ == '__main__':
    with open("../../data_set/AUDD/annotations/instances_train.json") as f:
        data = json.load(f)

    imgs = {}
    for img in data['images']:
        imgs[img['id']] = {
            'h': img['height'],
            'w': img['width'],
            'area': img['height'] * img['width'],
        }

    print(data['categories'])

    hw_ratios = []
    area_ratios = []
    label_count = defaultdict(int)
    for anno in data['annotations']:
        hw_ratios.append(anno['bbox'][3]/anno['bbox'][2])
        area_ratios.append(anno['area']/imgs[anno['image_id']]['area'])
        label_count[anno['category_id']] += 1

    # 从标签来看，总共多少个类别，如果加上背景类，总共多少个类别。
    # 各类别之间的框数量相对较平均，不需要调整默认的损失函数。（如果类别之间相差较大，建议调整损失函数，如BalancedL1Loss）
    # 平均每张图的框数量在4张左右，属于比较稀疏的检测，使用默认的keep_topk即可。
    # 一般多类别检测，建议使用MultiClassSoftNMS，而不是默认的MultiClassNMS，前者一般在多类别检测中效果较好。
    print(label_count, len(data['annotations']) / len(data['images']))

    # 查询长宽比
    # 这是真实框的宽高比，可以看到大部分集中在1.0左右，但也有部分在0.5~1之间，少部分在1.25~2.0之间。虽说anchor会进行回归得到更加准确的框，
    # 但是一开始给定一个相对靠近的anchor宽高比会让回归更加轻松。这里使用默认的 [0.5, 1, 2]即可。
    plt.hist(hw_ratios, bins=100, range=[0, 3])
    plt.show()

    # 查看
    # 这是真实框在原图的大小比例，可以看到大部分框只占到了原图的0.1%，甚至更小，因此基本都是很小的目标，这个也可以直接看一下原图和真实框就能发现。
    # 所以在初始的anchor_size设计时需要考虑到这一点，我这里anchor_size是从8开始的，也可以考虑从4开始，应该都可以的。
    plt.hist(area_ratios, bins=100, range=[0, 0.04])
    plt.show()