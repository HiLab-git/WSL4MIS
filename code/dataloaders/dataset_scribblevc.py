import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from random import sample

import pandas as pd

def pseudo_label_generator_prostate(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label =  segmentation - 1
    return pseudo_label


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label", train_dir="/Prostate_training_slices", val_dir="/Prostate_training_volumes"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.train_dir = train_dir
        self.val_dir = val_dir
        train_ids, test_ids = self._get_fold_ids(fold)
        self.catagory_list = pd.read_excel(self._base_dir + '/slice_classification.xlsx')
        # self.catagory_list = pd.read_excel(self._base_dir + '/increase_slice_classification.xlsx')
        self.catagory_list.set_index('slice', inplace=True)
        self.catagory_list = self.catagory_list.astype(bool)

        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + self.train_dir)
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + self.val_dir)
            self.sample_list = []
            print("test_ids", test_ids)
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 81)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        a_testing_set = [
            "patient{:0>3}".format(i) for i in [61,58,22,56,44,24,40,59,53,64,65,35,30,78,72,80,26,68,52,74]]
        a_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "a":
            return [a_training_set, a_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + self.train_dir +
                            "/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + self.val_dir +
                            "/{}".format(case), 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_prostate(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
            sample = {'image': image, 'label': label, 'gt': h5f['label'][:], 'category': torch.from_numpy( self.catagory_list.loc[case].values )}
            if self.transform:
                sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.int8)}
        sample["idx"] = case
        return sample


def random_rot_flip(image, label, gt):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    gt = np.rot90(gt, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    gt = np.flip(gt, axis=axis).copy()
    return image, label, gt


def random_rotate(image, label, gt, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="constant", cval=cval)
    gt = ndimage.rotate(gt, angle, order=0, reshape=False, mode="constant", cval=0)
    return image, label, gt


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, gt = sample['image'], sample['label'], sample['gt']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label, gt = random_rot_flip(image, label, gt)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label, gt = random_rotate(image, label, gt, cval=4)
            else:
                image, label, gt = random_rotate(image, label, gt, cval=0)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        gt = zoom(
            gt, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        gt = torch.from_numpy(gt.astype(np.uint8))
        sample['image'], sample['label'], sample['gt'] = image, label, gt
        return sample

class Zoom(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, gt = sample['image'], sample['label'], sample['gt']

        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        gt = zoom(
            gt, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        gt = torch.from_numpy(gt.astype(np.uint8))
        sample['image'], sample['label'], sample['gt'] = image, label, gt
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


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class ACDCDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label", train_dir="/ACDC_training_slices", val_dir="/ACDC_training_volumes"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.train_dir = train_dir
        self.val_dir = val_dir
        train_ids, test_ids = self._get_fold_ids(fold)
        self.catagory_list = pd.read_excel(self._base_dir + '/slice_classification.xlsx')
        # self.catagory_list = pd.read_excel(self._base_dir + '/increase_slice_classification.xlsx')
        self.catagory_list.set_index('slice', inplace=True)
        self.catagory_list = self.catagory_list.astype(bool)

        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + self.train_dir)
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + self.val_dir)
            self.sample_list = []
            print("test_ids", test_ids)
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        if fold == "MAAGfold":
            training_set = ["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90]]
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif fold == "MAAGfold70":
            training_set = ["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                               23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]]
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif "MAAGfold" in fold:
            training_num = int(fold[8:])
            training_set = sample(["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                               23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]], training_num)
            print("total {} training samples: {}".format(training_num, training_set))
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + self.train_dir +
                            "/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + self.val_dir +
                            "/{}".format(case), 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
            sample = {'image': image, 'label': label, 'gt': h5f['label'][:], 'category': torch.from_numpy( self.catagory_list.loc[case].values )}
            if self.transform:
                sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.int8)}
        sample["idx"] = case
        return sample


class MSCMRDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label",
                 train_dir="/MSCMR_training_slices", val_dir="/MSCMR_training_volumes"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.train_dir = train_dir
        self.val_dir = val_dir
        train_ids, test_ids = self._get_fold_ids(fold)
        self.catagory_list = pd.read_excel(self._base_dir + '/slice_classification.xlsx')
        # self.catagory_list = pd.read_excel(self._base_dir + '/increase_slice_classification.xlsx')
        self.catagory_list.set_index('slice', inplace=True)
        self.catagory_list = self.catagory_list.astype(bool)

        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + self.train_dir)
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + self.val_dir)
            self.sample_list = []
            print("test_ids", test_ids)
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        training_set = ["patient{:0>2}".format(i) for i in
                        [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7, 9]]
        validation_set = ["patient{:0>2}".format(i) for i in [1, 29, 36, 41, 8]]
        return [training_set, validation_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + self.train_dir +
                            "/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + self.val_dir +
                            "/{}".format(case), 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
            sample = {'image': image, 'label': label, 'gt': h5f['label'][:], 'category': torch.from_numpy( self.catagory_list.loc[case].values )}
            if self.transform:
                sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.int8)}
        sample["idx"] = case
        return sample
