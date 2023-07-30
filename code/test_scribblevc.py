import argparse
import importlib
import os
import re
import shutil

import SimpleITK as sitk
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.ndimage.interpolation import zoom
from torchmetrics.classification import MultilabelAccuracy
from tqdm import tqdm

from val_2D_scribblevc import calculate_metric_percase

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Prostate', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Prostate/Cross_Supervision_CRFLoss_DICE', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='conformer', help='model_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='fold')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')
parser.add_argument('--linear_layer', action="store_true", help='linear layer')
parser.add_argument('--bilinear', action="store_false", help='use bilinear in Upsample layers')
parser.add_argument('--predict_all', action="store_true", help='linear layer')


catagory_list = pd.read_excel('/home/zy/data/Prostate/slice_classification.xlsx')
catagory_list.set_index('slice', inplace=True)
catagory_list = catagory_list.astype(bool)
train_accuracy = MultilabelAccuracy(num_labels=2)


def get_fold_ids(fold):
    if fold == "fold1":
        return ["patient{:0>3}".format(i) for i in range(1, 21)]
    elif fold == "fold2":
        return ["patient{:0>3}".format(i) for i in range(21, 41)]
    elif fold == "fold3":
        return ["patient{:0>3}".format(i) for i in range(41, 61)]
    elif fold == "fold4":
        return ["patient{:0>3}".format(i) for i in range(61, 81)]
    else:
        return "ERROR KEY"

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/Prostate_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        category = torch.from_numpy( catagory_list.loc[case.replace(".h5", f"_slice_{ind}.h5")].values ).unsqueeze(0)
        # category = torch.from_numpy( catagory_list.loc[case.replace(".h5", f"_slice_{ind}.h5")].values ).unsqueeze(0).cuda()
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float()
            # 0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_aux1, out_aux2, _, cls_output, _ = net(input)
            out_aux1_soft = torch.softmax(out_aux1, dim=1)
            out_aux2_soft = torch.softmax(out_aux2, dim=1)
            out = torch.argmax(((0.5 * out_aux1_soft + 0.5 * out_aux2_soft)), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

            preds = 0.5 * cls_output[0] + 0.5 * cls_output[1]
            acc = train_accuracy(preds, category)

    total_train_acc = train_accuracy.compute()
    print(f"Accuracy on {case}: {total_train_acc}")
    train_accuracy.reset()
    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    ref_itk = sitk.ReadImage("../data/Prostate/all_prostate_volumes/{}.nii.gz".format(case.replace(".h5", "")))
    img_itk.CopyInformation(ref_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(ref_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(ref_itk)
    sitk.WriteImage(prd_itk, test_save_path + case.replace(".h5", "") + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case.replace(".h5", "") + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case.replace(".h5", "") + "_gt.nii.gz")
    return first_metric, second_metric, total_train_acc


def Inference(FLAGS):
    all_volumes = os.listdir(
        # FLAGS.root_path + "/cropped_slices")
        FLAGS.root_path + "/Prostate_training_volumes")
    if FLAGS.predict_all:
        image_list = all_volumes
    else:
        image_list = []
        test_ids = get_fold_ids(FLAGS.fold)
        for ids in test_ids:
            new_data_list = list(filter(lambda x: re.match(
                '{}.*'.format(ids), x) != None, all_volumes))
            image_list.extend(new_data_list)
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type)
    test_save_path = "../model/{}_{}/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = getattr(importlib.import_module("network.conformer_CAM"), 'Net_scribble_ACDC_gradCAM')\
        (linear_layer=FLAGS.linear_layer, bilinear=FLAGS.bilinear, num_classes=FLAGS.num_classes)

    save_mode_path=os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path,map_location='cpu'))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = []
    second_total = []
    train_acc_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, train_acc = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total.append(first_metric)
        second_total.append(second_metric)
        print(case, np.nanmean(np.array([first_metric[0],second_metric[0]])))
        train_acc_total += train_acc
    avg_metric = [np.nanmean(np.array(first_total), axis=0), np.nanmean(np.array(second_total), axis=0)]
    print(f"total acc: {train_acc_total/len(image_list)}")
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1])/2)
