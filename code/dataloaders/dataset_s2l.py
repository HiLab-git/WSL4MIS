import itertools
import os
import random
import re
from collections import defaultdict
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms


class BaseDataSets_s2l(Dataset):
    def __init__(self, base_dir=None, transform=None, fold="fold1", num=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.transform = transform
        train_ids, test_ids = self._get_fold_ids(fold)
        self.all_slices = os.listdir(self._base_dir + "/ACDC_training_slices")
        self.sample_list = []
        for ids in train_ids:
            new_data_list = list(filter(lambda x: re.match(
                '{}.*'.format(ids), x) != None, self.all_slices))
            self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))

        self.images = defaultdict(dict)
        for idx, case in enumerate(self.sample_list):
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')
            img = h5f['image']
            mask = h5f['label']
            scr = h5f['scribble']
            self.images[idx]['id'] = case
            self.images[idx]['image'] = np.array(img)
            self.images[idx]['mask'] = np.array(mask)
            self.images[idx]['scribble'] = np.array(scr)
            h, w = mask.shape
            self.images[idx]['weight'] = np.zeros((h, w, 4), dtype=np.float32)

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
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

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.images[idx]['id']
        image = self.images[idx]['image']
        mask = self.images[idx]['mask']
        scribble = self.images[idx]['scribble']
        weight = self.images[idx]['weight']
        sample = {'image': image, 'mask': mask,
                  'scribble': scribble, 'weight': weight}
        sample = self.transform(sample)
        sample['id'] = case
        return sample


def random_rot_flip(image, label, scribble, weight):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    scribble = np.rot90(scribble, k)
    weight = np.rot90(weight, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    scribble = np.flip(scribble, axis=axis).copy()
    weight = np.flip(weight, axis=axis).copy()
    return image, label, scribble, weight


def random_rotate(image, label, scribble, weight):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    scribble = ndimage.rotate(scribble, angle, order=0, reshape=False)
    weight = ndimage.rotate(weight, angle, order=0, reshape=False)
    return image, label, scribble, weight


class RandomGenerator_s2l(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, scribble, weight = sample['image'], sample['mask'], sample['scribble'], sample['weight']
        if random.random() > 0.5:
            image, label, scribble, weight = random_rot_flip(
                image, label, scribble, weight)
        elif random.random() > 0.5:
            image, label, scribble, weight = random_rotate(
                image, label, scribble, weight)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        scribble = zoom(
            scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        weight = zoom(
            weight, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        scribble = torch.from_numpy(scribble.astype(np.uint8))
        weight = torch.from_numpy(weight.astype(np.float32))
        sample = {'image': image, 'mask': label,
                  'scribble': scribble, 'weight': weight}
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


if __name__ == '__main__':
    data_root = '../data/ACDC/'
    labeled_slice = 146

    db_train = BaseDataSets(base_dir=data_root, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator([256, 256])]))
    db_val = BaseDataSets(base_dir=data_root, split="val")
    total_slices = len(db_train)
    labeled_slice = 146
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, 24, 24 - 12)

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    total_slices = len(db_train)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    print("{} iterations per epoch".format(len(trainloader)))

    for i_batch, sampled_batch in enumerate(trainloader):
        volume_batch, mask_batch, label_batch, pseudo_batch = sampled_batch[
            'image'], sampled_batch['mask'], sampled_batch['scribble'], sampled_batch['pseudo']
        case = sampled_batch['id'][:12]
        print(volume_batch.shape, mask_batch.shape,
              label_batch.shape, pseudo_batch.shape)
        print(case)
        print(torch.unique(mask_batch))