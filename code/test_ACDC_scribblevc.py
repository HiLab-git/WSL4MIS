import argparse
import os
import re
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

import importlib
from tool import pyutils
from time import strftime


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/scribbleVC', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='scribbleVC', help='model_name')
parser.add_argument('--fold', type=str,
                    default='MAAGfold', help='fold')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')
parser.add_argument("--network", default="network.scribbleVC", type=str)
parser.add_argument("--train_epochs", default="best", type=str)
parser.add_argument('--linear_layer', action="store_true", help='linear layer')
parser.add_argument('--bilinear', action="store_false", help='use bilinear in Upsample layers')
parser.add_argument('--save_prediction', action="store_true", help='save predictions while testing')
parser.add_argument("--arch", default='ACDC', type=str)


def get_fold_ids(fold):
    if fold == "MAAGfold":
        training_set = ["patient{:0>3}".format(i) for i in
                        [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                         71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90]]
        testing_set = ["patient{:0>3}".format(i) for i in
                          [5, 39, 77, 82, 78, 10, 64, 24, 30, 73, 80, 41, 36, 60, 72]]
        return [training_set, testing_set]
    elif "MAAGfold" in fold:
        training_set = ["patient{:0>3}".format(i) for i in
                        [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                         71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40,
                         7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                         23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]]
        testing_set = ["patient{:0>3}".format(i) for i in
                          [5, 39, 77, 82, 78, 10, 64, 24, 30, 73, 80, 41, 36, 60, 72]]
        return [training_set, testing_set]
    else:
        return "ERROR KEY"


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_aux1, out_aux2 = net(input)[0], net(input)[1]
            out_aux1_soft = torch.softmax(out_aux1, dim=1)
            out_aux2_soft = torch.softmax(out_aux2, dim=1)
            out = torch.argmax((out_aux1_soft+out_aux2_soft)*0.5, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    org_img_path = "../data/ACDC_training/{}.nii.gz".format(case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    if FLAGS.save_prediction:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        img_itk.CopyInformation(org_img_itk)
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk.CopyInformation(org_img_itk)
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        lab_itk.CopyInformation(org_img_itk)
        sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    train_ids, test_ids = get_fold_ids(FLAGS.fold)
    all_volumes = os.listdir(
        FLAGS.root_path + "/ACDC_training_volumes")
    image_list = []
    for ids in test_ids:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, all_volumes))
        image_list.extend(new_data_list)
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type)
    test_save_path = "../model/{}_{}/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)
    if FLAGS.save_prediction:
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path)
    logdir = os.path.join(test_save_path, "{}_log.txt".format(strftime("%Y_%m_%d_%H_%M_%S")))
    pyutils.Logger(logdir)
    print("log in ", logdir)

    net = getattr(importlib.import_module(FLAGS.network), 'scribbleVC_' + FLAGS.arch)(linear_layer=FLAGS.linear_layer, bilinear=FLAGS.bilinear)    # get Net_sm from network.conformer_CAM
    print('network is from', net.__class__)
    if FLAGS.train_epochs == "best":
        save_mode_path = os.path.join(
            snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    elif FLAGS.train_epochs == "final":
        save_mode_path = os.path.join(
            snapshot_path, '{}_final_model.pth'.format(FLAGS.model))
    else:
        save_mode_path = os.path.join(
            snapshot_path, '{}_{}_model.pth'.format(FLAGS.model, FLAGS.train_epochs))
    print("init weight from {}".format(save_mode_path))
    net.load_state_dict(torch.load(save_mode_path))

    net.eval()
    net.cuda()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        print(case)
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        print((first_metric[0] + second_metric[0] + third_metric[0]) / 3)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    print("RV:", avg_metric[0], " | Myo:", avg_metric[1], " | Lv:", avg_metric[2])
    print((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)
    return ((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)[0]


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    total = 0.0
    if FLAGS.fold == "all":
        for i in [1, 2, 3, 4, 5]:
            FLAGS.fold = "fold{}".format(i)
            print("Inference fold{}".format(i))
            mean_dice = Inference(FLAGS)
            print("Dice fold{}: {}".format(i, mean_dice))
            total += mean_dice
        print("mean dice of all 5-fold: ", total/5.0)
    else:
        print("Inference {}".format(FLAGS.fold))
        mean_dice = Inference(FLAGS)
        print("Dice {}: {}".format(FLAGS.fold, mean_dice))
