import itertools
import os
import random
import re
from glob import glob
import math
import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import logging


def read_txt(list_path):
    """读取txt文件中的内容"""
    sample_list = []
    with open(list_path, "r") as f: #按行读取txt中的图片名称
        for line in f.readlines():
            line = line.strip('\n')
            sample_list.append(line)  
    return sample_list


def pseudo_label_generator_acdc(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        markers[seed == 3] = 4
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label = segmentation - 1
    return pseudo_label
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, num=4, labeled_type="labeled", split='train', transform=None,  sup_type="label",label_ratio=0.1):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.num = num
        self.labeled_type = labeled_type
        self.label_ratio = label_ratio
        self.input_size = 256
        self.crop_size  = 128
        self.patch_num=1
        
        all_train_ids = read_txt("/mnt/sdd/tb/data/MSCMR/train_volumes.list")
        val_ids = read_txt("/mnt/sdd/tb/data/MSCMR/val_volumes.list")
        test_ids = read_txt("/mnt/sdd/tb/data/MSCMR/test_volumes.list")
        all_train_ids.sort()
        labeled_ids  = all_train_ids[:math.ceil(len(all_train_ids) * self.label_ratio)]
        # str_labeled_ids=labeled_ids[:len(labeled_ids)]  
 
        unlabeled_ids = [i for i in all_train_ids if i not in labeled_ids] 
        # unlabeled_ids = all_train_ids[int(len(all_train_ids) * self.label_ratio):]

        if self.split == 'train':
            self.all_slices = os.listdir(self._base_dir + "/MSCMR_training_slices")            
            self.sample_list = []

            if self.labeled_type == "labeled":
                print("Labeled patients IDs", labeled_ids)
                for ids in labeled_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids), x) != None, self.all_slices))
                    self.sample_list.extend(new_data_list)
                print("total labeled {} samples".format(len(self.sample_list)))
                logging.info("Unlabeled patients IDs:")
                logging.info(str(labeled_ids))

 
            if self.labeled_type == "unlabeled":
                print("Unlabeled patients IDs", unlabeled_ids)
                for ids in unlabeled_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids), x) != None, self.all_slices))
                    self.sample_list.extend(new_data_list)
                print("total unlabeled {} samples".format(len(self.sample_list)))
                logging.info("Unlabeled patients IDs:")
                logging.info(str(unlabeled_ids))


        if self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/MSCMR_training_volumes")
            self.sample_list = []
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        if self.split == 'test':
            self.all_volumes = os.listdir(
                self._base_dir + "/MSCMR_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +"/MSCMR_training_slices/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +"/MSCMR_training_volumes/{}".format(case), 'r')
        boxes = self.box_generation()
        
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            
            label_wr = pseudo_label_generator_acdc(image, h5f["scribble"][:])
            label = h5f['scribble'][:]
            sample = {'image': image, 'label': label,'random_walker':label_wr}
            sample = self.transform(sample) 
            
            crop_images = []
            for i in range(len(boxes)):
                box = boxes[i][1:]
                crop_images.append(sample['image'][:, box[1]:box[3], box[0]:box[2]].clone()[None])
            crop_images = torch.cat(crop_images, dim=0)           
            # crop_images=(sample['image'][:, box[1]:box[3], box[0]:box[2]].clone()[None])
            sample['boxes']=boxes
            sample['crop_images']=crop_images

        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            label[label==200]=2
            label[label==500]=3
            label[label==600]=1
            image=image.astype(np.float32)
            label=label.astype(np.uint8)
            # x, y = image.shape[1],image.shape[2]
            # image = zoom(image, (self.input_size[0] / x, self.input_size[1] / y), order=0)
            # label = zoom(label, (self.input_size[0] / x, self.input_size[1] / y), order=0)

            sample = {'image': image, 'label': label}
        sample["idx"] = case.split("_")[0]
        return sample

    def box_generation(self):
        max_range = self.input_size - self.crop_size
        boxes = []
        for i in range(self.patch_num):
            ind_h, ind_w = np.random.randint(0, max_range, size=2)
            boxes.append(torch.tensor([0, ind_w, ind_h, ind_w + self.crop_size, ind_h + self.crop_size])[None])
        boxes = torch.cat(boxes, dim=0)

        return boxes  # K, 5



def random_rot_flip(image, label,label_wr):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    label_wr = np.rot90(label_wr, k)

    axis = np.random.randint(0, 2)

    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    label_wr = np.flip(label_wr, axis=axis).copy()  

    return image, label,label_wr


def random_rotate(image, label,label_wr, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,reshape=False, mode="constant", cval=cval)
    label_wr = ndimage.rotate(label_wr, angle, order=0,reshape=False, mode="constant", cval=cval)
    return image, label,label_wr


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
                                #'random_walker':label_wr
    def __call__(self, sample):
        image, label ,label_wr= sample['image'], sample['label'], sample['random_walker']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label,label_wr = random_rot_flip(image, label,label_wr)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label,label_wr = random_rotate(image, label,label_wr, cval=4)
            else:
                image, label,label_wr = random_rotate(image, label,label_wr, cval=0)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_wr = zoom(label_wr, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        label_wr = torch.from_numpy(label_wr.astype(np.uint8))

        sample = {'image': image, 'label': label,'random_walker':label_wr}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
