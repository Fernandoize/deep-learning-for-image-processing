import json
import os
import random



import os
import shutil

import xmltodict

SRC_ROOT = "/Users/wangfengguo/LocalTools/data/DFUIDataSet"
VOC_ROOT = os.path.join(SRC_ROOT, "VOCdevkit", "VOC2012")

import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

def create_voc_xml(real_name, image_info, annotations, categories):
    # 创建根元素
    annotation = Element('annotation')

    # 添加图像信息
    folder = SubElement(annotation, 'folder')
    folder.text = 'VOC2012'
    filename = SubElement(annotation, 'filename')
    filename.text = real_name + ".jpg"
    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(image_info['width'])
    height = SubElement(size, 'height')
    height.text = str(image_info['height'])
    depth = SubElement(size, 'depth')
    depth.text = '3'

    # 添加标注信息
    for ann in annotations:
        obj = SubElement(annotation, 'object')
        name = SubElement(obj, 'name')
        name.text = categories[ann['category_id']]
        difficult = SubElement(obj, 'difficult')
        difficult.text = str(ann['iscrowd'])
        bbox = SubElement(obj, 'bndbox')
        xmin = SubElement(bbox, 'xmin')
        xmin.text = str(int(ann['bbox'][0]))
        ymin = SubElement(bbox, 'ymin')
        ymin.text = str(int(ann['bbox'][1]))
        xmax = SubElement(bbox, 'xmax')
        xmax.text = str(int(ann['bbox'][0] + ann['bbox'][2]))
        ymax = SubElement(bbox, 'ymax')
        ymax.text = str(int(ann['bbox'][1] + ann['bbox'][3]))

    # 格式化XML
    xml_str = tostring(annotation, 'utf-8')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

    return pretty_xml

def save_voc_xml(xml_str, output_dir, image_id):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    xml_file = os.path.join(output_dir, f'{image_id}.xml')
    with open(xml_file, 'w') as f:
        f.write(xml_str)


def create_voc_structure(voc_root):
    voc_dirs = [
        os.path.join(voc_root, 'Annotations'),
        os.path.join(voc_root, 'ImageSets', 'Main'),
        os.path.join(voc_root, 'JPEGImages')
    ]

    # 1. 创建所有目录
    for d in voc_dirs:
        os.makedirs(d, exist_ok=True)

def main():
    create_voc_structure(VOC_ROOT)

    voc_annotation_folder = os.path.join(VOC_ROOT, "Annotations")
    voc_image_folder = os.path.join(VOC_ROOT, "JPEGImages")
    voc_imageset_main_folder = os.path.join(VOC_ROOT, 'ImageSets', 'Main')

    random.seed(0)  # 设置随机种子，保证随机结果可复现
    for data_type in ['train', 'val', 'test']:
        annotation_file = f'{SRC_ROOT}/annotations/instances_{data_type}2017.json'
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # 创建类别ID到类别名称的映射
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        class_json = {v:k for k, v in categories.items()}
        print(class_json)

        # 遍历所有图像
        file_names = set()
        src_images_folder = os.path.join(SRC_ROOT, "images")
        for index, image_info in enumerate(coco_data['images']):
            image_id = image_info['id']
            image_name = image_info['file_name']
            annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
            # 这里filename使用类型+id,防止冲突
            file_name = f"{data_type}_{image_id}"
            if file_name in file_names:
                raise Exception("file name conflict")
            file_names.add(file_name)
            xml_str = create_voc_xml(file_name, image_info, annotations, categories)
            # 以image_id命名
            save_voc_xml(xml_str, voc_annotation_folder, f"{file_name}")
            shutil.copy(os.path.join(src_images_folder, image_name), os.path.join(VOC_ROOT, voc_image_folder, f"{file_name}.jpg"))
            print(f"{data_type}: {index + 1}/{len(coco_data['images'])}", end="\r")

        try:
            f = open(f"{voc_imageset_main_folder}/{data_type}.txt", "w")
            f.write("\n".join(file_names))
        except FileExistsError as e:
            print(e)
            exit(1)
        print(f"{data_type}.txt total file: {len(file_names)}")


if __name__ == '__main__':
    main()

