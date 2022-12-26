# -*- coding: utf-8 -*-
# Author: Shuojue Yang (main contribution) and Xiangde Luo (minor modification for WORD and other datasets).
# Date:   16 Dec. 2021
# Implementation for simulation of the sparse scribble annotation based on the dense annotation for the WORD dataset and other datasets.
# # Reference:
# @article{luo2022scribbleseg,
# title={Scribble-Supervised Medical Image Segmentation via Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision},
# author={Xiangde Luo, Minhao Hu, Wenjun Liao, Shuwei Zhai, Tao Song, Guotai Wang, Shaoting Zhang},
# journal={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
# year={2022},
# pages={528--538}}

# @article{luo2022word,
# title={{WORD}: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image},
# author={Xiangde Luo, Wenjun Liao, Jianghong Xiao, Jieneng Chen, Tao Song, Xiaofan Zhang, Kang Li, Dimitris N. Metaxas, Guotai Wang, and Shaoting Zhang},
# journal={Medical Image Analysis},
# volume={82},
# pages={102642},
# year={2022},
# publisher={Elsevier}}

# @misc{wsl4mis2020,
# title={{WSL4MIS}},
# author={Luo, Xiangde},
# howpublished={\url{https://github.com/Luoxd1996/WSL4MIS}},
# year={2021}}
# If you have any questions, please contact Xiangde Luo (https://luoxd1996.github.io).


import glob
import math
import random
import sys

import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

sys.setrecursionlimit(1000000)
seed = 2022
np.random.seed(seed)
random.seed(seed)


def random_rotation(image, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    img = Image.fromarray(image)
    img_rotate = img.rotate(angle)
    return img_rotate


def translate_img(img, x_shift, y_shift):

    (height, width) = img.shape[:2]
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    trans_img = cv2.warpAffine(img, matrix, (width, height))
    return trans_img


def get_largest_two_component_2D(img, print_info=False, threshold=None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 2D volume
        threshold: a size threshold
    outputs:
        out_img: the output volume 
    """
    s = ndimage.generate_binary_structure(2, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches+1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(print_info):
        print('component size', sizes_list)
    if(len(sizes) == 1):
        out_img = [img]
    else:
        if(threshold):
            max_size1 = sizes_list[-1]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            if max_label1.shape[0] > 1:
                max_label1 = max_label1[0]
            component1 = labeled_array == max_label1
            out_img = [component1]
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab[0]
                    out_img.append(temp_cmp)
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            if max_label1.shape[0] > 1:
                max_label1 = max_label1[0]
            if max_label2.shape[0] > 1:
                max_label2 = max_label2[0]
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(max_size2*10 > max_size1):
                out_img = [component1, component2]
            else:
                out_img = [component1]
    return out_img


class Cutting_branch(object):
    def __init__(self):
        self.lst_bifur_pt = 0
        self.branch_state = 0
        self.lst_branch_state = 0
        self.direction2delta = {0: [-1, -1], 1: [-1, 0], 2: [-1, 1], 3: [
            0, -1], 4: [0, 0], 5: [0, 1], 6: [1, -1], 7: [1, 0], 8: [1, 1]}

    def __find_start(self, lab):
        y, x = lab.shape
        idxes = np.asarray(np.nonzero(lab))
        for i in range(idxes.shape[1]):
            pt = tuple([idxes[0, i], idxes[1, i]])
            assert lab[pt] == 1
            directions = []
            for d in range(9):
                if d == 4:
                    continue
                if self.__detect_pt_bifur_state(lab, pt, d):
                    directions.append(d)
            if len(directions) == 1:
                start = pt
                self.start = start
                self.output[start] = 1
                return start
        start = tuple([idxes[0, 0], idxes[1, 0]])
        self.output[start] = 1
        self.start = start
        return start

    def __detect_pt_bifur_state(self, lab, pt, direction):

        d = direction
        y = pt[0] + self.direction2delta[d][0]
        x = pt[1] + self.direction2delta[d][1]
        if lab[y, x] > 0:
            return True
        else:
            return False

    def __detect_neighbor_bifur_state(self, lab, pt):
        directions = []
        for i in range(9):
            if i == 4:
                continue
            if self.output[tuple([pt[0] + self.direction2delta[i][0], pt[1] + self.direction2delta[i][1]])] > 0:
                continue
            if self.__detect_pt_bifur_state(lab, pt, i):
                directions.append(i)

        if len(directions) == 0:
            self.end = pt
            return False
        else:
            direction = random.sample(directions, 1)[0]
            next_pt = tuple([pt[0] + self.direction2delta[direction]
                            [0], pt[1] + self.direction2delta[direction][1]])
            if len(directions) > 1 and pt != self.start:
                self.lst_output = self.output*1
                self.previous_bifurPts.append(pt)
            self.output[next_pt] = 1
            pt = next_pt
            self.__detect_neighbor_bifur_state(lab, pt)

    def __detect_loop_branch(self, end):
        for d in range(9):
            if d == 4:
                continue
            y = end[0] + self.direction2delta[d][0]
            x = end[1] + self.direction2delta[d][1]
            if (y, x) in self.previous_bifurPts:
                self.output = self.lst_output * 1
                return True

    def __call__(self, lab, seg_lab, iterations=1):
        self.previous_bifurPts = []
        self.output = np.zeros_like(lab)
        self.lst_output = np.zeros_like(lab)
        components = get_largest_two_component_2D(lab, threshold=15)
        if len(components) > 1:
            for c in components:
                start = self.__find_start(c)
                self.__detect_neighbor_bifur_state(c, start)
        else:
            c = components[0]
            start = self.__find_start(c)
            self.__detect_neighbor_bifur_state(c, start)
        self.__detect_loop_branch(self.end)
        struct = ndimage.generate_binary_structure(2, 2)
        output = ndimage.morphology.binary_dilation(
            self.output, structure=struct, iterations=iterations)
        shift_y = random.randint(-6, 6)
        shift_x = random.randint(-6, 6)
        if np.sum(seg_lab) > 1000:
            output = translate_img(output.astype(np.uint8), shift_x, shift_y)
            output = random_rotation(output)
        output = output * seg_lab
        return output


def scrible_2d(label, iteration=[4, 10]):
    lab = label
    skeleton_map = np.zeros_like(lab, dtype=np.int32)
    for i in range(lab.shape[0]):
        if np.sum(lab[i]) == 0:
            continue
        struct = ndimage.generate_binary_structure(2, 2)
        if np.sum(lab[i]) > 900 and iteration != 0 and iteration != [0] and iteration != None:
            iter_num = math.ceil(
                iteration[0]+random.random() * (iteration[1]-iteration[0]))
            slic = ndimage.morphology.binary_erosion(
                lab[i], structure=struct, iterations=iter_num)
        else:
            slic = lab[i]
        sk_slice = skeletonize(slic, method='lee')
        sk_slice = np.asarray((sk_slice == 255), dtype=np.int32)
        skeleton_map[i] = sk_slice
    return skeleton_map


def scribble4class(label, class_id, class_num, iteration=[4, 10], cut_branch=True):
    label = (label == class_id)
    sk_map = scrible_2d(label, iteration=iteration)
    if cut_branch and class_id != 0:
        cut = Cutting_branch()
        for i in range(sk_map.shape[0]):
            lab = sk_map[i]
            if lab.sum() < 1:
                continue
            sk_map[i] = cut(lab, seg_lab=label[i])
    if class_id == 0:
        class_id = class_num
    return sk_map * class_id


def generate_scribble(label, iterations, cut_branch=True):
    class_num = np.max(label) + 1
    output = np.zeros_like(label, dtype=np.uint8)
    for i in range(class_num):
        it = iterations[i] if isinstance(iterations, list) else iterations
        scribble = scribble4class(
            label, i, class_num, it, cut_branch=cut_branch)
        output += scribble.astype(np.uint8)
    return output


if __name__ == "__main__":
    num = 0
    for i in sorted(glob.glob("../imgs/*_lab.nii.gz")):
        print("{} Begin".format(i.split("/")[-1]))
        itk_data = sitk.ReadImage(i)
        label = sitk.GetArrayFromImage(itk_data)
        num_classes = 3  # total segmentation classes
        output = generate_scribble(label, tuple([1, num_classes-1]))
        # ignore index for partially cross-entropy loss
        output[output == 0] = 255
        output[output == num_classes] = 0
        itk_scr = sitk.GetImageFromArray(output)
        itk_scr.CopyInformation(itk_data)
        sitk.WriteImage(itk_scr, i.replace('_lab.nii.gz', '_scribble.nii.gz'))
        print("{} End".format(i.split("/")[-1]))
        print(num)
        num += 1
