import os
from collections import defaultdict

from pytorch_object_detection.duo.voc_convert import read_xml_info

ROOT_DIR = "/Users/wangfengguo/LocalTools/data/DUODataSet"
VOC_ROOT = os.path.join(ROOT_DIR, "VOCdevkit", "VOC2012")


result_dict = {
    'train': [],
    'val': [],
    'test': []
}

def stats(voc_root):
    txt_folder = os.path.join(voc_root, "ImageSets", "Main")
    for data_type in ["train", "val", "test"]:
        with open(os.path.join(txt_folder, f"{data_type}.txt"), "r") as f:
            file_names = f.readlines()
            for file_name in file_names:
                annotation_file = os.path.join(voc_root, 'Annotations', file_name.strip() + ".xml")
                image_info = read_xml_info(annotation_file)
                result_dict[data_type].append(image_info)

    for data_type, images_info in result_dict.items():
        count = len(images_info)
        object_stats = defaultdict(int)
        image_size = []
        difficult = 0
        for image_info in images_info:
            object_count = len(image_info['objects'])
            object_stats[object_count] += 1

            for object in image_info['objects']:
                if int(object['difficult']) > 0:
                    difficult += 1
                image_size.append(f"{image_info['height']}-{image_info['width']}-{image_info['depth']}")
        object_stats = dict(sorted(object_stats.items()))
        print(f"{data_type}_size: {count}, object_stats: {object_stats}, difficult={difficult}, image_size={set(image_size)}")

        # with open(f'{data_type}_stats.csv', 'w') as f:
        #     f.write(','.join([str(_) for _ in list(object_stats.keys())]))
        #     f.write('\n')
        #     f.write(','.join([str(_) for _ in list(object_stats.values())]))


if __name__ == '__main__':
    stats(VOC_ROOT)