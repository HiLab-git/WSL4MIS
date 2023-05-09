import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn.functional as F
import cv2
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, device,patch_size=[256, 256]):
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
                0).unsqueeze(0).float().to(device)
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
            0).unsqueeze(0).float().to(device)
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

def test_single_volume2(image, label, net, classes, device,patch_size=[256, 256]):
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
                0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                # output=F.interpolate(net(input)[0], size=label.shape[1:], mode='bilinear', align_corners=False)
                out=net(input)[0]
                # out=F.interpolate(out, size=patch_size, mode='bilinear', align_corners=False)
                out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input)[0], dim=1), dim=1).squeeze(0)
            out=F.interpolate(out, size=label.shape[1:], mode='bilinear', align_corners=False)
            prediction = out.cpu().detach().numpy()
                        
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list



def test_single_volume_ds(image, label, net, classes, device,patch_size=[256, 256]):
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
                0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                output_main, _, _, _ = net(input)
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_cct(image, label, net, classes, device,patch_size=[256, 256]):
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
                0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2) # 计算图像对角线长度
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
        
    mask = mask.astype(np.uint8)
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    
    # 因为之前向四周填充了0, 故而这里不再需要四周
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def calculate_metric_percase_7(y_pred, y_true):
    """ pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        if gt.sum() > 0:
            hd = metric.binary.hd(pred, gt)
        else:
            hd = 0

        eps = 0.0001
        dilation_ratio=0.005
        c_pred, h_pred, w_pred = pred.shape
        y_pred, y_true = np.array(pred), np.array(gt)
        y_pred, y_true = np.round(pred).astype(int), np.round(gt).astype(int)
        a_unin_b = np.sum(y_pred[y_true == 1])
        a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
        # dice
        #dice_value = (a_unin_b * 2.0 + eps) / a_plus_b
        # PPV
        ppv_value = (a_unin_b * 1.0 + eps) / (np.sum(y_pred) + eps)

        # sensitivity
        sen_val = (a_unin_b * 1.0 + eps) / (np.sum(y_true) + eps)
        #print('ppv_value and sen_val has been calculated')
        iou = a_unin_b / (a_plus_b - a_unin_b) # a_plus_b里边有eps，所以不加了
        
        boundary_iou_all = 0.0
        for i in range(c_pred):
            gt_boundary = mask_to_boundary(y_true[i], dilation_ratio)
            dt_boundary = mask_to_boundary(y_pred[i], dilation_ratio)
            intersection = ((gt_boundary * dt_boundary) > 0).sum()
            union = ((gt_boundary + dt_boundary) > 0).sum()
            boundary_iou = intersection / (union + eps)
            boundary_iou_all += boundary_iou
        boundary_iou = boundary_iou_all / c_pred """
    eps = 0.0001
    c_pred, h_pred, w_pred = y_pred.shape
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    TP = np.sum(y_pred[y_true == 1])

    a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
    #dice
    dice=(TP * 2.0 + eps) / a_plus_b
    denominator1=np.sum(y_pred) + eps
    #PPV
    ppv=(TP*1.0 + eps) / denominator1
    denominator2 = np.sum(y_true) + eps
    #Sen
    sen=(TP*1.0 + eps) / denominator2
    
    # hd and hd95
    if y_pred.sum() > 0 and y_true.sum() > 0:
        hd = metric.binary.hd(y_pred, y_true)
        hd95 = metric.binary.hd95(y_pred, y_true)
        asd  = metric.binary.asd(y_pred, y_true)
    else:
        hd = 0
        hd95 = 0
        asd = 0
    # iou
    a_unin_b = np.sum(y_pred[y_true == 1]) + eps
    a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
    iou = a_unin_b / (a_plus_b - a_unin_b)

    # biou
    boundary_iou_all = 0.0
    dilation_ratio=0.005
    for i in range(c_pred):
        gt_boundary = mask_to_boundary(y_true[i], dilation_ratio)
        dt_boundary = mask_to_boundary(y_pred[i], dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        boundary_iou = intersection / (union + eps)
        boundary_iou_all += boundary_iou
    boundary_iou = boundary_iou_all / c_pred 

    return dice, hd95, ppv, sen, iou, boundary_iou, hd,asd

def test_single_volume_7(image, label, net, classes, device,patch_size=[256, 256],):
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
                0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input)[0], dim=1), dim=1).squeeze(0)
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
                net(input)[0], dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_7(
            prediction == i, label == i))

    performance_test = np.mean(metric_list, axis=0)[0]
    # if is_save_img==True:
    #         save_imgs_rgb(out_path,data_name,img_np,mask_np,pred_mask,patientSliceID,performance_test,exp)

    return metric_list
