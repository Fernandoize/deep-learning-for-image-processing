import json
import os.path

import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


# 加载COCO数据集
def load_coco_data(json_path):
    coco = COCO(json_path)
    ann_ids = coco.getAnnIds()
    annotations = coco.loadAnns(ann_ids)
    return annotations


# 计算中心点到四个边的距离
def calculate_distances(bbox):
    # bbox格式: [x, y, width, height]
    x, y, w, h = bbox

    # 计算中心点坐标
    center_x = x + w / 2
    center_y = y + h / 2

    # 到四个边的距离
    left_dist = center_x - x
    right_dist = (x + w) - center_x
    top_dist = center_y - y
    bottom_dist = (y + h) - center_y

    return [left_dist, right_dist, top_dist, bottom_dist]


# 处理所有标注并统计分布
def process_annotations(annotations):
    all_distances = {
        'left': [],
        'right': [],
        'top': [],
        'bottom': []
    }

    for ann in annotations:
        if 'bbox' in ann:
            distances = calculate_distances(ann['bbox'])
            all_distances['left'].append(distances[0])
            all_distances['right'].append(distances[1])
            all_distances['top'].append(distances[2])
            all_distances['bottom'].append(distances[3])

    return all_distances


# 绘制分布图
def plot_distributions(dtype, distances_dict):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 四个边的标签和对应子图位置
    titles = ['Left Distance', 'Right Distance', 'Top Distance', 'Bottom Distance']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, (key, pos) in enumerate(zip(distances_dict.keys(), positions)):
        axs[pos].hist(distances_dict[key], bins=50, density=True)
        axs[pos].set_title(f'{dtype}-{titles[i]}')
        axs[pos].set_xlabel(f'Distance (pixels)')
        axs[pos].set_ylabel('Density')

    plt.tight_layout()
    plt.show()


SRC_ROOT = "../../data_set/AUDD"

# 主函数
def main():
    # 替换为你的COCO annotation文件路径
    for type in ['train', 'val', 'test']:
        json_path = os.path.join(SRC_ROOT, 'annotations', f'instances_{type}.json')

        if not os.path.exists(json_path):
            continue

        # 加载数据
        print("加载COCO数据...")
        annotations = load_coco_data(json_path)

        # 处理标注
        print("计算距离分布...")
        distances_dict = process_annotations(annotations)

        # 绘制分布图
        print("绘制分布图...")
        plot_distributions(type, distances_dict)

        # 输出基本统计信息
        for key in distances_dict:
            print(f"\n{key} distance statistics:")
            print(f"Mean: {np.mean(distances_dict[key]):.2f}")
            print(f"Std: {np.std(distances_dict[key]):.2f}")
            print(f"Min: {np.min(distances_dict[key]):.2f}")
            print(f"Max: {np.max(distances_dict[key]):.2f}")


if __name__ == "__main__":
    main()