import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple, TypeVar

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

A = TypeVar("A")
B = TypeVar("B")


def is_image_file(filename: str) -> bool:
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(data_root: str,
                 data_list: str,
                 class_list: List[int],
                 remove_images_with_undesired_classes: bool = False,
                 keep_small_area_classes: bool = False) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    """
        Recovers all tuples (img_path, label_path) relevant to the current experiments (class_list
        is used as filter)

        input:
            data_root : Path to the data directory
            data_list : Path to the .txt file that contain the train/test split of images
            class_list: List of classes to keep
        returns:
            image_label_list: List of (img_path, label_path) that contain at least 1 object of a class
                              in class_list
            class_file_dict: Dict of all (img_path, label_path that contain at least 1 object of a class
                              in class_list, grouped by classes.
    """
    image_label_list: List[Tuple[str, str]] = []
    list_read = open(data_list).readlines()

    print(f"Processing data for {class_list}")
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    process_partial = partial(process_image, data_root=data_root, class_list=class_list,
                              remove_images_with_undesired_classes=remove_images_with_undesired_classes,
                              keep_small_area_classes=keep_small_area_classes)

    with Pool(os.cpu_count() // 2) as pool:
        for sublist, subdict in pool.map(process_partial, tqdm(list_read)):  # mmap
            image_label_list += sublist
            for (k, v) in subdict.items():
                class_file_dict[k] += v
        pool.close()
        pool.join()

    return image_label_list, class_file_dict


def process_image(line: str,
                  data_root: str,
                  class_list: List,
                  remove_images_with_undesired_classes: bool,
                  keep_small_area_classes: bool) -> Tuple[List, Dict]:
    """
        Reads and parses a line corresponding to 1 file

        input:
            line : A line corresponding to 1 file, in the format path_to_image.jpg path_to_image.png
            data_root : Path to the data directory
            class_list: List of classes to keep

    """
    line = line.strip()
    line_split = line.split(' ')
    image_name = os.path.join(data_root, line_split[0])
    label_name = os.path.join(data_root, line_split[1])
    item: Tuple[str, str] = (image_name, label_name)
    label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
    label_class = np.unique(label).tolist()

    if 0 in label_class:
        label_class.remove(0)
    if 255 in label_class:
        label_class.remove(255)
    for label_class_ in label_class:
        assert label_class_ in list(range(1, 81)), label_class_

    c: int
    new_label_class = []
    for c in label_class:
        if c in class_list:
            tmp_label = np.zeros_like(label)
            target_pix = np.where(label == c)
            tmp_label[target_pix[0], target_pix[1]] = 1
            if tmp_label.sum() >= 16 * 32 * 32 or keep_small_area_classes:
                new_label_class.append(c)
        elif remove_images_with_undesired_classes:
            new_label_class = []
            break

    label_class = new_label_class

    image_label_list: List[Tuple[str, str]] = []
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    if len(label_class) > 0:
        image_label_list.append(item)

        for c in label_class:
            assert c in class_list
            class_file_dict[c].append(item)

    return image_label_list, class_file_dict
