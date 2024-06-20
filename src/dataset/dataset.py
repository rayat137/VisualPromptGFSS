import argparse
import random
from multiprocessing import Pool
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os

try:
    import src.dataset.transform as transform
except ImportError:
    import dataset.transform as transform
#import dataset.transform as transform #src.dataset.transform #as transform
from .classes import get_split_classes
from .utils import make_dataset
from .classes import classId2className



def get_val_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """
        Build the validation loader.
    """
    assert args.split in [0, 1, 2, 3, 10, 11, -1]
    val_transform = transform.Compose([
            transform.Resize(args.image_size),
            transform.ToTensor(),
            transform.Normalize(mean=args.mean, std=args.std)])
    split_classes = get_split_classes(args)

    # ===================== Get base and novel classes =====================
    print(f'Data: {args.data_name}, S{args.split}')
    base_class_list = split_classes[args.data_name][args.split]['train']
    novel_class_list = split_classes[args.data_name][args.split]['val']
    print('Novel classes:', novel_class_list)
    args.num_classes_tr = len(base_class_list) + 1  # +1 for bg
    args.num_classes_val = len(novel_class_list)

    # ===================== Build loader =====================
    val_sampler = None
    if args.multi_way:
        torch.manual_seed(1)
        val_data = MultiClassValData(transform=val_transform,
                                    base_class_list=base_class_list,
                                    novel_class_list=novel_class_list,
                                    data_list_path_train=args.train_list,
                                    data_list_path_test=args.val_list,
                                    args=args)
        val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size_val,
                                             drop_last=True,
                                             shuffle=args.shuffle_test_data,
                                             num_workers=args.workers,
                                             pin_memory=args.pin_memory,
                                             sampler=val_sampler)
    else:
        args.num_classes_val = 1
        val_data = ClassicValData(transform=val_transform,
                                    base_class_list=base_class_list,
                                    novel_class_list=novel_class_list,
                                    data_list_path_train=args.train_list,
                                    data_list_path_test=args.val_list,
                                    args=args)
        
        val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1,
                                             drop_last=True,
                                             shuffle=args.shuffle_test_data,
                                             num_workers=args.workers,
                                             pin_memory=args.pin_memory,
                                             sampler=val_sampler)
    
    return val_loader, val_data


def get_image_and_label(image_path, label_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
    return image, label


def adjust_label(base_class_list, novel_class_list, label, chosen_novel_class,  base_label=-1, other_novels_label=255):
    # -1 for base_label or other_novels_label means including the true labels
    assert base_label in [-1, 0, 255] and other_novels_label in [-1, 0, 255]
    new_label = np.zeros_like(label)  # background
    #import pdb; pdb.set_trace()
    for lab in base_class_list:
        indexes = np.where(label == lab)
        if base_label == -1:
            new_label[indexes[0], indexes[1]] = base_class_list.index(lab) + 1  # Add 1 because class 0 is bg
        else:
            new_label[indexes[0], indexes[1]] = base_label

    for lab in novel_class_list:
        indexes = np.where(label == lab)
        if other_novels_label == -1:
            new_label[indexes[0], indexes[1]] = 1 + len(base_class_list) + novel_class_list.index(lab)
        elif lab == chosen_novel_class:
            new_label[indexes[0], indexes[1]] = 1 + len(base_class_list)
        else:
            new_label[indexes[0], indexes[1]] = other_novels_label

    ignore_pix = np.where(label == 255)
    new_label[ignore_pix] = 255

    return new_label


class ClassicValData(Dataset):
    def __init__(self, transform: transform.Compose, base_class_list: List[int], novel_class_list: List[int],
                 data_list_path_train: str, data_list_path_test: str, args: argparse.Namespace):
        assert args.support_only_one_novel
        self.shot = args.shot
        self.data_root = args.data_root
        self.base_class_list = base_class_list
        self.novel_class_list = novel_class_list
        self.transform = transform

        self.use_training_images_for_supports = args.use_training_images_for_supports
        assert not self.use_training_images_for_supports or data_list_path_train
        support_data_list_path = data_list_path_train if self.use_training_images_for_supports else data_list_path_test
        
        if os.path.exists('split0_query.npy'):
            np_dict = np.load('split0_query.npy', allow_pickle=True).item()
            self.query_data_list = np_dict['query_list']
        else:
            self.query_data_list, _ = make_dataset(args.data_root, data_list_path_test,
                                                self.base_class_list + self.novel_class_list,
                                                keep_small_area_classes=True)
            print('Total number of kept images (query):', len(self.query_data_list))
            np_dict = {'query_list': self.query_data_list}
            np.save('split0_query.npy', np_dict)

        
        if os.path.exists('split0_support.npy'):
            np_dict = np.load('split0_support.npy', allow_pickle=True).item()
            self.support_data_list = np_dict['support_data_list']
            self.support_sub_class_file_list  = np_dict['support_sub_cls_list']
        else:
            self.support_data_list, self.support_sub_class_file_list = make_dataset(args.data_root, support_data_list_path,
                                                                                    self.novel_class_list,
                                                                                    keep_small_area_classes=False)
            
            np_dict = {'support_data_list': self.support_data_list, 'support_sub_cls_list': self.support_sub_class_file_list}
            np.save('split0_support.npy', np_dict)

        print('Total number of kept images (support):', len(self.support_data_list))

    @property
    def num_novel_classes(self):
        return len(self.novel_class_list)

    @property
    def all_classes(self):
        return [0] + self.base_class_list + self.novel_class_list

    def _adjust_label(self, label, chosen_novel_class,  base_label=-1, other_novels_label=255):
        return adjust_label(self.base_class_list, self.novel_class_list,
                            label, chosen_novel_class,  base_label, other_novels_label)

    def __len__(self):
        return len(self.query_data_list)

    def __getitem__(self, index):
        # ========= Read query image and Choose class =======================
        image_path, label_path = self.query_data_list[index]
        qry_img, label = get_image_and_label(image_path, label_path)
        if self.transform is not None:
            qry_img, label = self.transform(qry_img, label)

        # == From classes in the query image, choose one randomly ===
        label_class = set(np.unique(label))
        label_class -= {0, 255}
        novel_classes_in_image = list(label_class.intersection(set(self.novel_class_list)))
        if len(novel_classes_in_image) > 0:
            class_chosen = np.random.choice(novel_classes_in_image)
        else:
            class_chosen = np.random.choice(self.novel_class_list)

        q_valid_pixels = (label != 255).float()
        target = self._adjust_label(label, class_chosen, base_label=-1, other_novels_label=0)

        support_image_list = []
        support_label_list = []

        file_class_chosen = self.support_sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        # ========= Build support ==============================================
        # == First, randomly choose indexes of support images ==
        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []

        for _ in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            while (support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list:
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        # == Second, read support images and masks  ============
        for k in range(self.shot):
            support_image_path, support_label_path = support_image_path_list[k], support_label_path_list[k]
            support_image, support_label = get_image_and_label(support_image_path, support_label_path)
            support_label = self._adjust_label(support_label, class_chosen, base_label=0, other_novels_label=0)
            support_image_list.append(support_image)
            support_label_list.append(support_label)

        # == Forward images through transforms =================
        if self.transform is not None:
            for k in range(len(support_image_list)):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k], support_label_list[k])
                support_image_list[k] = support_image_list[k].unsqueeze(0)
                support_label_list[k] = support_label_list[k].unsqueeze(0)

        # == Reshape properly ==================================
        spprt_imgs = torch.cat(support_image_list, 0)
        spprt_labels = torch.cat(support_label_list, 0)

        return qry_img, target, q_valid_pixels, spprt_imgs, spprt_labels, class_chosen


class MultiClassValData(Dataset):
    def __init__(self, transform: transform.Compose, base_class_list: List[int], novel_class_list: List[int],
                 data_list_path_train: str, data_list_path_test: str, args: argparse.Namespace):
        self.support_only_one_novel = args.support_only_one_novel
        self.use_training_images_for_supports = args.use_training_images_for_supports
        assert not self.use_training_images_for_supports or data_list_path_train
        support_data_list_path = data_list_path_train if self.use_training_images_for_supports else data_list_path_test
        # if args.shot>5:
        #     self.shot=5
        # else:
        #     self.shot = args.shot
        self.class_names = classId2className
        self.data_name = args.data_name
        self.shot = args.shot
        self.data_root = args.data_root
        self.base_class_list = base_class_list  # Does not contain bg
        self.novel_class_list = novel_class_list
        if self.data_name == 'pascal':
            split_query_list = 'pascal_split{0}_query.npy'.format(args.split)
            split_support_list = 'pascal_split{0}_support.npy'.format(args.split)
        else:
            split_query_list = 'split{0}_query.npy'.format(args.split)
            split_support_list = 'split{0}_support.npy'.format(args.split)

        ## FIX SEED FOR RANDOM
        random.seed(1)
        if os.path.exists(split_query_list):
            np_dict = np.load(split_query_list, allow_pickle=True).item()
            self.query_data_list = np_dict['query_list']

        else:
            self.query_data_list, _ = make_dataset(args.data_root, data_list_path_test,
                                                self.base_class_list + self.novel_class_list,
                                                keep_small_area_classes=True)
            
            np_dict = {'query_list': self.query_data_list}
            np.save(split_query_list, np_dict)

        
        self.complete_query_data_list = self.query_data_list.copy()
        print('Total number of kept images (query):', len(self.query_data_list))


        if os.path.exists(split_support_list):
            np_dict = np.load(split_support_list, allow_pickle=True).item()
            support_data_list = np_dict['support_data_list']
            self.support_sub_class_file_list  = np_dict['support_sub_cls_list']

        else:
            support_data_list, self.support_sub_class_file_list = make_dataset(args.data_root, support_data_list_path,
                                                                           self.novel_class_list,
                                                                           keep_small_area_classes=False)
            
            np_dict = {'support_data_list': support_data_list, 'support_sub_cls_list': self.support_sub_class_file_list}
            np.save(split_support_list, np_dict)
        print('Total number of kept images (support):', len(support_data_list))
        self.transform = transform

    @property
    def num_novel_classes(self):
        return len(self.novel_class_list)

    @property
    def all_classes(self):
        return [0] + self.base_class_list + self.novel_class_list

    def _adjust_label(self, label, chosen_novel_class, base_label=-1, other_novels_label=255):
        return adjust_label(self.base_class_list, self.novel_class_list,
                            label, chosen_novel_class, base_label, other_novels_label)

    def __len__(self):
        return len(self.query_data_list)

    def __getitem__(self, index):  # It only gives the query
        image_path, label_path = self.query_data_list[index]
        qry_img, label = get_image_and_label(image_path, label_path)
        label = self._adjust_label(label, -1, base_label=-1, other_novels_label=-1)
        if self.transform is not None:
            #import pdb; pdb.set_trace()
            qry_img, label = self.transform(qry_img, label)
        valid_pixels = (label != 255).float()
        return qry_img, label, valid_pixels, image_path
    def get_base_class_names(self):
        class_names = ['background']
        print(self.base_class_list)
        for i in self.base_class_list:
            class_names.append(self.class_names[self.data_name][i])

        return class_names
    
    def get_novel_class_names(self):
        class_names = []

        for i in self.novel_class_list:
            class_names.append(self.class_names[self.data_name][i])

        return class_names

    def generate_support(self, query_image_path_list, remove_them_from_query_data_list=False):
        image_list, label_list, original_label_list = list(), list(), list()
        support_image_path_list, support_label_path_list = list(), list()
        for c in self.novel_class_list:
            file_class_chosen = self.support_sub_class_file_list[c]
            num_file = len(file_class_chosen)
            indices_list = list(range(num_file))
            random.shuffle(indices_list)
            current_path_list = list()
            for idx in indices_list:
                if len(current_path_list) >= self.shot:
                    break
                image_path, label_path = file_class_chosen[idx]
                if image_path in (query_image_path_list + current_path_list):
                    continue
                image, label = get_image_and_label(image_path, label_path)
                if self.support_only_one_novel:  # Ignore images that have multiple novel classes
                    present_novel_classes = set(np.unique(label)) - {0, 255} - set(self.base_class_list)
                    if len(present_novel_classes) > 1:
                        continue
                original_label = self._adjust_label(label, -1, base_label=-1, other_novels_label=-1)
                label = self._adjust_label(label, -1, base_label=0, other_novels_label=-1)  # If support_only_one_novel is True, images with more than one novel classes won't reach this line. So, -1 won't make the image contain two different novel classes.
                
                image_list.append(image)
                label_list.append(label)
                original_label_list.append(original_label)
                current_path_list.append(image_path)
                support_image_path_list.append(image_path)
                support_label_path_list.append(label_path)
            found_images_count = len(current_path_list)
            assert found_images_count > 0, f'No support candidate for class {c} out of {num_file} images'
            if found_images_count < self.shot:
                indices_to_repeat = random.choices(range(found_images_count), k=self.shot-found_images_count)
                image_list.extend([image_list[i] for i in indices_to_repeat])
                label_list.extend([label_list[i] for i in indices_to_repeat])
                original_label_list.extend([original_label_list[i] for i in indices_to_repeat])

        transformed_image_list, transformed_label_list, transformed_original_label_list = list(), list(), list()
        if self.shot == 1:
            for i, l, o in zip(image_list, label_list, original_label_list):
                #import pdb; pdb.set_trace()
                transformed_i, transformed_l, transformed_o = self.transform(i, l, o)
                transformed_image_list.append(transformed_i.unsqueeze(0))
                transformed_label_list.append(transformed_l.unsqueeze(0))
                transformed_original_label_list.append(transformed_o.unsqueeze(0))
        else:
            with Pool(self.shot) as pool:
                for transformed_i, transformed_l, transformed_o in pool.starmap(self.transform, zip(image_list, label_list, original_label_list)):
                    transformed_image_list.append(transformed_i.unsqueeze(0))
                    transformed_label_list.append(transformed_l.unsqueeze(0))
                    transformed_original_label_list.append(transformed_o.unsqueeze(0))
                pool.close()
                pool.join()
        #import pdb; pdb.set_trace()
        spprt_imgs = torch.cat(transformed_image_list, 0)
        spprt_labels = torch.cat(transformed_label_list, 0)
        spprt_labels_original = torch.cat(transformed_original_label_list, 0)

        if remove_them_from_query_data_list and not self.use_training_images_for_supports:
            self.query_data_list = self.complete_query_data_list.copy()
            for i, l in zip(support_image_path_list, support_label_path_list):
                self.query_data_list.remove((i, l))
        return spprt_imgs, spprt_labels, spprt_labels_original


# -------------------------- Pre-Training --------------------------

class BaseData(Dataset):
    def __init__(self, split=3, mode=None, data_root=None, data_list=None, data_set=None, use_split_coco=False, transform=None, main_process=False, \
                batch_size=None, mask_classification=False):

        assert data_set in ['pascal', 'coco']
        assert mode in ['train', 'val']

        if data_set == 'pascal':
            self.num_classes = 20
        elif data_set == 'coco':
            self.num_classes = 80
        self.dataset = data_set
        self.mode = mode
        self.split = split 
        self.data_root = data_root
        self.batch_size = batch_size
        self.mask_classification = mask_classification
        self.class_names = classId2className
        if data_set == 'pascal':
            self.class_list = list(range(1, 21))                         # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3: 
                self.sub_list = list(range(1, 16))                       # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21))                  # [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16))                  # [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11))                   # [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21))                       # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6))                    # [1,2,3,4,5]

        elif data_set == 'coco':
            if use_split_coco:
                print('INFO: using SPLIT COCO (FWB)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            else:
                print('INFO: using COCO (PANet)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))           

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    

        self.data_list = []  
        list_read = open(data_list).readlines()
        print("Processing data...")

        for l_idx in tqdm(range(len(list_read))):
            line = list_read[l_idx]
            line = line.strip()
            line_split = line.split(' ')
            image_name = os.path.join(self.data_root, line_split[0])
            label_name = os.path.join(self.data_root, line_split[1])
            item = (image_name, label_name)
            self.data_list.append(item)

        self.transform = transform

    def get_class_names(self):
        class_names = ['background']

        for i in self.sub_list:
            class_names.append(self.class_names[self.dataset][i])

        return class_names
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_tmp = label.copy()
        
        k_hot_vector  = torch.zeros(len(self.sub_list)+1)

        for cls in range(1, self.num_classes+1):
            select_pix = np.where(label_tmp == cls)
            if cls in self.sub_list:
                label[select_pix[0],select_pix[1]] =  self.sub_list.index(cls) + 1
            else:
                label[select_pix[0],select_pix[1]] = 0

                
        raw_label = label.copy()

        if self.transform is not None:
            image, label = self.transform(image, label)

        binary_mask_gt = torch.zeros(len(self.sub_list)+1, label.shape[0], label.shape[1], dtype=torch.int64)

        classes_present = torch.unique(label)
        classes_present = classes_present[classes_present!=0]
        classes_present = classes_present[classes_present!=255]

        for c in classes_present:
            binary_mask_gt[c, : ,:] = (label == c).type(torch.int64)
            k_hot_vector[c] = 1

        

        #print(k_hot_vector.shape)
        # Return
        if self.mode == 'val' and self.batch_size == 1:
            return image, label, raw_label
        
        elif self.mask_classification:
            label = binary_mask_gt
            return image, (label, k_hot_vector)
        else:
            return image, label