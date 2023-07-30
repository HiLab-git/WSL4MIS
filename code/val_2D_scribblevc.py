import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import pandas as pd
from torchmetrics.classification import MultilabelAccuracy


catagory_list = pd.read_excel('../data/ACDC/slice_classification.xlsx')
catagory_list.set_index('slice', inplace=True)
catagory_list = catagory_list.astype(bool)
test_accuracy = MultilabelAccuracy(num_labels=4).cuda()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if gt.sum() == 0 and pred.sum() == 0:
        return np.nan, np.nan
    elif gt.sum() == 0 and pred.sum() > 0:
        return 0, 0
    elif gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if pred.sum() == 0:
            hd95 = np.nan
        else:
            hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input), dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


@torch.no_grad()
def test_single_volume_CAM(image, label, net, classes, patch_size=[256, 256], epoch=None, model_type=None):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output = net(input, ep=epoch, model_type=model_type)
                out_aux1, out_aux2, cls_output = output
                out_aux1_soft = torch.softmax(out_aux1, dim=1)
                out_aux2_soft = torch.softmax(out_aux2, dim=1)
                out = torch.argmax(((torch.min(out_aux1_soft, out_aux2_soft) > 0.5) * \
                                    (0.5 * out_aux1_soft + 0.5 * out_aux2_soft)), dim=1).squeeze(0)

                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred

    else:
        input = torch.from_numpy(image).float().cuda()

        net.eval()
        with torch.no_grad():
            out_aux1, out_aux2 = net(input)[0], net(input)[1]
            out_aux1_soft = torch.softmax(out_aux1, dim=1)
            out_aux2_soft = torch.softmax(out_aux2, dim=1)
            out = torch.argmax(((torch.min(out_aux1_soft, out_aux2_soft) > 0.5) * \
                                (0.5 * out_aux1_soft + 0.5 * out_aux2_soft)), dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            prediction = zoom(
                out, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))

    return metric_list
