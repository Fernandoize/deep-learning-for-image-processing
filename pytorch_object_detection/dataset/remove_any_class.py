import json
import os.path


def remove_category_from_coco(input_json_path, output_json_path, category_name='person'):
    # 读取COCO JSON文件
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # 获取要删除的category_id
    category_id = None
    for category in coco_data['categories']:
        if category['name'] == category_name:
            category_id = category['id']
            break

    if category_id is None:
        print(f"未找到类别 {category_name}")
        return

    # 过滤掉指定类别的annotations
    original_count = len(coco_data['annotations'])
    coco_data['annotations'] = [ann for ann in coco_data['annotations']
                                if ann['category_id'] != category_id]

    removed_count = original_count - len(coco_data['annotations'])

    # 保存新的JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)

    print(f"已完成清理，从 {original_count} 个标注中移除了 {removed_count} 个 {category_name} 类的标注")

SRC_ROOT = "../../data_set/dfui"

if __name__ == '__main__':
    # 使用示例
    for type in ['train', 'val', 'test']:
        input_json_path = os.path.join(SRC_ROOT, 'annotations', f'instances_{type}2017.json')
        output_json_path = os.path.join(SRC_ROOT, 'annotations', f'instances_{type}2017_output.json')
        remove_category_from_coco(input_json_path, output_json_path, 'waterweeds')