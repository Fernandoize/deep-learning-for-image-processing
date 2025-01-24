import os
import random



import os
import shutil

import xmltodict

ROOT_DIR = "/Users/wangfengguo/LocalTools/data/DUODataSet"
VOC_ROOT = os.path.join(ROOT_DIR, "VOCdevkit", "VOC2012")

def create_voc_structure(voc_root, image_folder):
    voc_dirs = [
        os.path.join(voc_root, 'Annotations'),
        os.path.join(voc_root, 'ImageSets', 'Main'),
        os.path.join(voc_root, 'JPEGImages')
    ]

    # 1. 创建所有目录
    for d in voc_dirs:
        os.makedirs(d, exist_ok=True)

    # 2. 生成train.txt, val.txt. test.txt
    for data_type in ['train', 'val', 'test']:
        data_folder = os.path.join(image_folder, data_type)
        files_names = sorted([file.rsplit(".", 1)[0] for file in os.listdir(data_folder) if file.endswith(".jpg")])
        files_num = len(files_names)

        txt_folder = os.path.join(voc_root, 'ImageSets', 'Main')
        try:
            f = open(f"{txt_folder}/{data_type}.txt", "w")
            f.write("\n".join(files_names))
        except FileExistsError as e:
            print(e)
            exit(1)
        print(f"{data_type}.txt total file: {files_num}")

        # 移动/复制图片到 JPEGImages 文件夹
        voc_images_folder = os.path.join(voc_root, 'JPEGImages')
        for index, filename in enumerate(files_names):
            shutil.copy(os.path.join(image_folder, data_type, filename + ".jpg"), voc_images_folder)

        voc_annotation_folder = os.path.join(voc_root, 'Annotations')
        for index, filename in enumerate(files_names):
            shutil.copy(os.path.join(image_folder, data_type, filename + ".xml"), voc_annotation_folder)

    image_count = len(os.listdir(os.path.join(voc_root, 'JPEGImages')))
    annotation_count = len(os.listdir(os.path.join(voc_root, 'Annotations')))
    print(f"total images: {image_count}, total annotations: {annotation_count}")


def generate_classes(voc_root):
    total_empty_objects = 0
    annotation_folder = os.path.join(voc_root, 'Annotations')
    object_name = []
    for file in os.listdir(annotation_folder):
        image_info = read_xml_info(os.path.join(annotation_folder, file))
        name = [o['name'] for o in image_info['objects']]
        object_name.extend(name)

    print(set(object_name))


def read_xml_info(xml_path):
    """
    解析xml文件，返回类别名
    """
    with open(xml_path, 'rb') as f:
        xml_dict = xmltodict.parse(f)
        annotations = xml_dict['annotation']
        objects = annotations.get('object', [])
        if isinstance(objects, dict):
            objects = [objects]
        return {
            'height': annotations['size']['height'],
            'width': annotations['size']['width'],
            'depth': annotations['size']['depth'],
            'objects': objects
        }


def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现
    # create_voc_structure(VOC_ROOT, ROOT_DIR)
    generate_classes(VOC_ROOT)


if __name__ == '__main__':
    main()

